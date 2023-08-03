import os
import random
import quaternion
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image


class LocalizationDataset(Dataset):
    def __init__(self, data_path='./dataset', img_size=(1080, 1080), split='train', negatives=2, 
                 projection_mode='undefined', visible_kernel_size=9, dtype=torch.float32, 
                 verbose=True, seed=0, pointcloud_points=65536):
        super().__init__()
        # important args
        self.data_path = data_path
        self.img_size = img_size
        self.split = split
        self.num_negatives = negatives
        self.projection_mode = projection_mode
        self.visible_kernel_size = visible_kernel_size
        self.dtype = dtype

        # minor args
        self.verbose = verbose
        self.seed = seed
        self.pointcloud_points = pointcloud_points # only for empty positives / negatives

        assert self.split in ['train', 'test', 'test_only']
        assert self.projection_mode in ['undefined', 'min']
        # undefined: fast, but non-closest point might be selected for projection
        # min: slow, but guaranteed to project the closest point
        assert self.pointcloud_points > 0

        # CAUTION: resize, crop changes intrinsic!
        if self.img_size is None:
            transforms_list = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        elif isinstance(self.img_size, int):
            transforms_list = [transforms.ToPILImage(), transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        else:
            transforms_list = [transforms.ToPILImage(), transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(transforms_list)

        self.database = [] # [(pc_path, seq)]
        self.database_seq = torch.zeros((0,)) # torch.Tensor([seq_idx])
        self.query = [] # [(img_path, pc_path, 4*4 pose array, intrinsic, seq)]

        # for visibility check
        self.maxpool = torch.nn.MaxPool2d(self.visible_kernel_size, stride=1, padding=self.visible_kernel_size // 2)
        self.minpool = lambda x: -self.maxpool(-x)

        random.seed(self.seed)


    def tqdm(self, x, desc='', leave=True, total=None):
        if self.verbose:
            return tqdm(x, desc=desc, leave=leave, total=total)
        else:
            return x


    def print(self, x):
        if self.verbose:
            print(x)
    

    def load_train(self, folder='files'):
        self.database = [] # [(pc_path, seq)]
        self.database_seq = [] # [seq_idx]
        self.query = [] # [(img_path, pc_path, 4*4 pose array, intrinsic, seq)]
        for seq_idx, seq in self.tqdm(enumerate(self.seq), desc='seq', leave=True, total=len(self.seq)):
            with open(os.path.join(self.data_path, folder, '%s_%s.txt'%(seq, self.split)), 'r') as f:
                lines = f.readlines()
            for line in self.tqdm(lines, desc='files', leave=False):
                line = line.split()
                img_path, pc_path = line[:2]
                pose_intrinsic = [float(x) for x in line[2:]]
                img_pose = np.array(pose_intrinsic[:16]).reshape((4, 4))
                img_intrinsic = np.array(pose_intrinsic[16:25]).reshape((3, 3))
                self.database.append((pc_path, seq))
                self.database_seq.append(seq_idx)
                self.query.append((img_path, pc_path, img_pose, img_intrinsic, seq))
        self.database_seq = torch.tensor(self.database_seq)
    

    def load_test(self, folder='files'):
        self.database = [] # [(pc_path, seq_]
        self.database_seq = [] # [seq_idx]
        self.query = [] # [(img_path, pc_path, 4*4 pose array, intrinsic, seq)]
        for seq_idx, seq in self.tqdm(enumerate(self.seq), desc='seq', leave=True, total=len(self.seq)):
            lines = []
            mode = 'query_pc'
            if os.path.isfile(os.path.join(self.data_path, folder, '%s_train.txt'%seq)):
                with open(os.path.join(self.data_path, folder, '%s_train.txt'%seq), 'r') as f:
                    lines = f.readlines()
            if len(lines) > 0:
                mode = 'database_pc'
                for line in lines:
                    pc_path = line.split()[1]
                    self.database.append((pc_path, seq))
                    self.database_seq.append(seq_idx)
            with open(os.path.join(self.data_path, folder, '%s_test.txt'%seq), 'r') as f:
                lines = f.readlines()
            for line in self.tqdm(lines, desc='files', leave=False):
                line = line.split()
                img_path, pc_path = line[:2]
                pose_intrinsic = [float(x) for x in line[2:]]
                img_pose = np.array(pose_intrinsic[:16]).reshape((4, 4))
                img_intrinsic = np.array(pose_intrinsic[16:25]).reshape((3, 3))
                if mode == 'query_pc':
                    self.database.append((pc_path, seq))
                    self.database_seq.append(seq_idx)
                self.query.append((img_path, pc_path, img_pose, img_intrinsic, seq))
        self.database_seq = torch.tensor(self.database_seq)
    

    def load(self, folder='files'):
        if self.split == 'train':
            self.load_train(folder)
        elif self.split in ['test', 'test_only']:
            self.load_test(folder)
    

    def load_pc(self, pc_path):
        if pc_path[-4:] == '.bin':
            return torch.tensor(np.fromfile(pc_path, dtype=np.float64).reshape((-1, 3)), dtype=self.dtype)
        elif pc_path[-4:] == '.npy':
            return torch.tensor(np.load(pc_path), dtype=self.dtype)
        else:
            raise NotImplementedError


    def get_negative_idx(self, idx):
        seq_idx = self.database_seq[idx]
        # (with replacement) return random.choices(list(torch.nonzero(self.database_seq != seq_idx, as_tuple=True)[0]), k=self.num_negatives)
        return random.sample(list(torch.nonzero(self.database_seq != seq_idx, as_tuple=True)[0]), k=self.num_negatives)


    def get_idx_train(self, idx):
        img_path, pc_path, img_pose, img_intrinsic, img_seq = self.query[idx]
        img = np.array(Image.open(img_path))
        oh, ow = img.shape[:2]
        img = self.transform(img).to(dtype=self.dtype)
        h, w = img.shape[1:]
        img_pose_quat = torch.cat([torch.tensor(img_pose[:3, 3]), torch.tensor(quaternion.as_float_array(quaternion.from_rotation_matrix(img_pose[:3, :3]))).to(dtype=self.dtype)], dim=-1)
        img_pose = torch.tensor(img_pose).to(dtype=self.dtype)
        img_intrinsic = torch.tensor(img_intrinsic).to(dtype=self.dtype)
        img_intrinsic[0] *= (w / ow)
        img_intrinsic[1] *= (h / oh)

        positives = self.load_pc(pc_path)
        positive_coords = positives.mean(0)
        positives -= positive_coords.view(1, 3)
        #positives = ME.SparseTensor(features=torch.ones((self.pointcloud_points, 1), dtype=torch.float32), 
        #                            coordinates=positives)
        
        if self.num_negatives == 0:
            negatives = torch.zeros((self.num_negatives, self.pointcloud_points, 3), dtype=self.dtype)
            negative_coords = torch.zeros((self.num_negatives, 3), dtype=self.dtype)
            negative_images = torch.zeros((self.num_negatives,) + img.shape, dtype=self.dtype)
            negative_images_pose = torch.zeros((self.num_negatives, 4, 4), dtype=self.dtype)
            negative_images_pose_quat = torch.zeros((self.num_negatives, 7), dtype=self.dtype)
            negative_images_intrinsic = torch.zeros((self.num_negatives, 3, 3), dtype=self.dtype)
        else:
            neg_idx_list = self.get_negative_idx(idx)
            negatives = torch.stack([self.load_pc(self.database[neg_idx][0]) for neg_idx in neg_idx_list], dim=0)
            negative_coords = negatives.mean(1)
            negatives -= negative_coords.view(self.num_negatives, 1, 3)

            negative_images = []
            negative_images_pose = []
            negative_images_pose_quat = []
            negative_images_intrinsic = []
            for i, neg_idx in enumerate(neg_idx_list):
                neg_path, _, neg_pose, neg_intrinsic, _ = self.query[neg_idx]
                neg_img = np.array(Image.open(neg_path))
                oh, ow = neg_img.shape[:2]
                neg_img = self.transform(neg_img).to(dtype=self.dtype)
                h, w = neg_img.shape[1:]
                neg_pose_quat = torch.cat([torch.tensor(neg_pose[:3, 3]), torch.tensor(quaternion.as_float_array(quaternion.from_rotation_matrix(neg_pose[:3, :3]))).to(dtype=self.dtype)], dim=-1)
                neg_pose = torch.tensor(neg_pose).to(dtype=self.dtype)
                neg_intrinsic = torch.tensor(neg_intrinsic).to(dtype=self.dtype)
                neg_intrinsic[0] *= (w / ow)
                neg_intrinsic[1] *= (h / oh)
                negative_images.append(neg_img)
                negative_images_pose.append(neg_pose)
                negative_images_pose_quat.append(neg_pose_quat)
                negative_images_intrinsic.append(neg_intrinsic)
                #negatives[i] = ME.SparseTensor(features=torch.ones((self.pointcloud_points, 1), dtype=torch.float32), 
                #                               coordinates=negatives[i])
            negative_images = torch.stack(negative_images)
            negative_images_pose = torch.stack(negative_images_pose)
            negative_images_pose_quat = torch.stack(negative_images_pose_quat)
            negative_images_intrinsic = torch.stack(negative_images_intrinsic)

        # depth = z value (not distance)
        depth_map = -torch.ones(1, img.shape[1], img.shape[2]).to(dtype=self.dtype) # -1: unknown
        pc = positives + positive_coords.view(1, 3) # N, 3
        pc = torch.cat([pc, torch.ones_like(pc[:, :1])], dim=1) # N, 4
        pc_in_cam = img_pose @ torch.t(pc) # 4, N
        pc_in_cam = pc_in_cam[:3] / pc_in_cam[3:] # 3, N
        pc_in_img = img_intrinsic @ pc_in_cam # 3, N
        pc_xy = pc_in_img[:2] / pc_in_img[2:] # 2, N
        pc_xy = torch.round(pc_xy.to(dtype=torch.float32)).to(dtype=torch.int64)
        pc_ind = torch.all(torch.stack([pc_in_img[2] > 0, 0 <= pc_xy[0], pc_xy[0] <= w - 1, 0 <= pc_xy[1], pc_xy[1] <= h - 1], dim=0), dim=0)
        if self.projection_mode == 'undefined':
            depth_map[0, pc_xy[1, pc_ind], pc_xy[0, pc_ind]] = pc_in_cam[2, pc_ind] # depth might corrupted for the points with same image coordinates
        elif self.projection_mode == 'min':
            pc_depth = pc_in_cam[2, pc_ind]
            depth_argsort = torch.argsort(pc_depth)
            pc_xy = pc_xy[:, pc_ind][:, depth_argsort]
            pc_depth = pc_depth[depth_argsort]
            _, unique_indices = np.unique(np.array(pc_xy), return_index=True, axis=1)
            depth_map[0, pc_xy[1, unique_indices], pc_xy[0, unique_indices]] = pc_depth[unique_indices]
        
        visible_depth_map = depth_map.clone().detach()
        visible_depth_map[visible_depth_map == -1] = float('inf')
        visible_depth_map[self.maxpool(self.minpool(visible_depth_map)) < visible_depth_map] = float('inf')
        visible_depth_map[visible_depth_map == float('inf')] = -1

        data = {
            'image': img,
            #'image_path': img_path,
            #'image_pose': img_pose,
            'image_pose_quat': img_pose_quat, # (x, y, z, qw, qx, qy, qz)
            'img_intrinsic': img_intrinsic,
            'positives': positives,
            'negatives': negatives,
            #'negative_images': negative_images,
            #'negative_images_pose': negative_images_pose,
            #'negative_images_pose_quat': negative_images_pose_quat,
            #'negative_images_intrinsic': negative_images_intrinsic,
            'positive_coords': positive_coords,
            'negative_coords': negative_coords,
            'depth_map': depth_map,
            'visible_depth_map': visible_depth_map,
        }
        return data


    def get_idx_test(self, idx):
        if idx < len(self.database):
            pc_path, img_seq = self.database[idx]
            img_path = None
            img = None
            img_pose = None
            img_intrinsic = None
        else:
            idx -= len(self.database)
            img_path, pc_path, img_pose, img_intrinsic, img_seq = self.query[idx]
            img = np.array(Image.open(img_path))
            oh, ow = img.shape[:2]
            img = self.transform(img).to(dtype=self.dtype)
            h, w = img.shape[1:]
            img_pose = torch.tensor(img_pose).to(dtype=self.dtype)
            img_intrinsic = torch.tensor(img_intrinsic).to(dtype=self.dtype)
            img_intrinsic[0] *= (w / ow)
            img_intrinsic[1] *= (h / oh)

        positives = self.load_pc(pc_path)
        positive_coords = positives.mean(0)
        positives -= positive_coords.view(1, 3)

        if img is None:
            depth_map = None
            visible_depth_map = None
        else:
            # depth = z value (not distance)
            depth_map = -torch.ones(1, img.shape[1], img.shape[2]).to(dtype=self.dtype) # -1: unknown
            pc = positives + positive_coords.view(1, 3) # N, 3
            pc = torch.cat([pc, torch.ones_like(pc[:, :1])], dim=1) # N, 4
            pc_in_cam = img_pose @ torch.t(pc) # 4, N
            pc_in_cam = pc_in_cam[:3] / pc_in_cam[3:] # 3, N
            pc_in_img = img_intrinsic @ pc_in_cam # 3, N
            pc_xy = pc_in_img[:2] / pc_in_img[2:] # 2, N
            pc_xy = torch.round(pc_xy.to(dtype=torch.float32)).to(dtype=torch.int64)
            pc_ind = torch.all(torch.stack([pc_in_img[2] > 0, 0 <= pc_xy[0], pc_xy[0] <= w - 1, 0 <= pc_xy[1], pc_xy[1] <= h - 1], dim=0), dim=0)
            if self.projection_mode == 'undefined':
                depth_map[0, pc_xy[1, pc_ind], pc_xy[0, pc_ind]] = pc_in_cam[2, pc_ind] # depth might corrupted for the points with same image coordinates
            elif self.projection_mode == 'min':
                pc_depth = pc_in_cam[2, pc_ind]
                depth_argsort = torch.argsort(pc_depth)
                pc_xy = pc_xy[:, pc_ind][:, depth_argsort]
                pc_depth = pc_depth[depth_argsort]
                _, unique_indices = np.unique(np.array(pc_xy), return_index=True, axis=1)
                depth_map[0, pc_xy[1, unique_indices], pc_xy[0, unique_indices]] = pc_depth[unique_indices]
            
            visible_depth_map = depth_map.clone().detach()
            visible_depth_map[visible_depth_map == -1] = float('inf')
            visible_depth_map[self.maxpool(self.minpool(visible_depth_map)) < visible_depth_map] = float('inf')
            visible_depth_map[visible_depth_map == float('inf')] = -1

        data = {
            'image': img,
            'image_path': img_path,
            'image_pose': img_pose,
            'img_intrinsic': img_intrinsic,
            'positives': positives,
            'positive_coords': positive_coords,
            'depth_map': depth_map,
            'visible_depth_map': visible_depth_map,
            'sequence': img_seq,
        }
        return data


    def get_idx_test_only(self, idx):
        img_path, pc_path, img_pose, img_intrinsic, img_seq = self.query[idx]
        img = np.array(Image.open(img_path))
        oh, ow = img.shape[:2]
        img = self.transform(img).to(dtype=self.dtype)
        h, w = img.shape[1:]
        img_pose = torch.tensor(img_pose).to(dtype=self.dtype)
        img_intrinsic = torch.tensor(img_intrinsic).to(dtype=self.dtype)
        img_intrinsic[0] *= (w / ow)
        img_intrinsic[1] *= (h / oh)

        positives = self.load_pc(pc_path)
        positive_coords = positives.mean(0)
        positives -= positive_coords.view(1, 3)

        # depth = z value (not distance)
        depth_map = -torch.ones(1, img.shape[1], img.shape[2]).to(dtype=self.dtype) # -1: unknown
        pc = positives + positive_coords.view(1, 3) # N, 3
        pc = torch.cat([pc, torch.ones_like(pc[:, :1])], dim=1) # N, 4
        pc_in_cam = img_pose @ torch.t(pc) # 4, N
        pc_in_cam = pc_in_cam[:3] / pc_in_cam[3:] # 3, N
        pc_in_img = img_intrinsic @ pc_in_cam # 3, N
        pc_xy = pc_in_img[:2] / pc_in_img[2:] # 2, N
        pc_xy = torch.round(pc_xy.to(dtype=torch.float32)).to(dtype=torch.int64)
        pc_ind = torch.all(torch.stack([pc_in_img[2] > 0, 0 <= pc_xy[0], pc_xy[0] <= w - 1, 0 <= pc_xy[1], pc_xy[1] <= h - 1], dim=0), dim=0)
        if self.projection_mode == 'undefined':
            depth_map[0, pc_xy[1, pc_ind], pc_xy[0, pc_ind]] = pc_in_cam[2, pc_ind] # depth might corrupted for the points with same image coordinates
        elif self.projection_mode == 'min':
            pc_depth = pc_in_cam[2, pc_ind]
            depth_argsort = torch.argsort(pc_depth)
            pc_xy = pc_xy[:, pc_ind][:, depth_argsort]
            pc_depth = pc_depth[depth_argsort]
            _, unique_indices = np.unique(np.array(pc_xy), return_index=True, axis=1)
            depth_map[0, pc_xy[1, unique_indices], pc_xy[0, unique_indices]] = pc_depth[unique_indices]
        
        visible_depth_map = depth_map.clone().detach()
        visible_depth_map[visible_depth_map == -1] = float('inf')
        visible_depth_map[self.maxpool(self.minpool(visible_depth_map)) < visible_depth_map] = float('inf')
        visible_depth_map[visible_depth_map == float('inf')] = -1

        data = {
            'image': img,
            'image_path': img_path,
            'image_pose': img_pose,
            'img_intrinsic': img_intrinsic,
            'positives': positives,
            'positive_coords': positive_coords,
            'depth_map': depth_map,
            'visible_depth_map': visible_depth_map,
            'sequence': img_seq,
        }
        return data


    def __len__(self):
        if self.split in ['train', 'test_only']:
            return len(self.query)
        elif self.split == 'test':
            return len(self.database) + len(self.query)
    
    
    def __getitem__(self, idx):
        if self.split == 'train':
            return self.get_idx_train(idx)
        elif self.split == 'test':
            return self.get_idx_test(idx)
        elif self.split == 'test_only':
            return self.get_idx_test_only(idx)
        else:
            return None
    


class KITTI(LocalizationDataset):
    # image coordinate (x, y) = (dim1, dim0)
    # camera_pose: world to cam
    # lidar_pose: world coordinate of center of lidar
    # img(x, y) = intrinsic @ camera_pose @ (pointcloud + pointcloud_pose)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.seq_all = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        if self.split == 'train':
            self.seq = self.seq_all[:9]
        elif self.split == 'test':
            self.seq = self.seq_all[-2:]
        else:
            raise NotImplementedError

        self.load('files')



class s2d3ds(LocalizationDataset):
    # image coordinate (x, y) = (dim1, dim0)
    # camera_pose: world to cam
    # lidar_pose: world coordinate of center of lidar
    # img(x, y) = intrinsic @ camera_pose @ (pointcloud + pointcloud_pose)
    def __init__(self, area_split='12346_5', split_train_angle=30, split_test_angle=15, **kwargs):
        super().__init__(**kwargs)
        self.area_split = area_split
        self.split_train_angle = split_train_angle
        self.split_test_angle = split_test_angle

        assert self.area_split in ['12346_5', '1356_24', '245_136']

        self.seq_all = ['1', '2', '3', '4', '5', '6']
        if self.split == 'train':
            self.seq = []
            for seq in self.area_split.split('_')[0]:
                self.seq.append(seq)
        elif self.split == 'test':
            self.seq = []
            for seq in self.area_split.split('_')[-1]:
                self.seq.append(seq)
        else:
            raise NotImplementedError
        
        self.load('files_' + self.area_split + '_%d_%d'%(self.split_train_angle, self.split_test_angle))

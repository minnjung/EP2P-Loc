import os
import glob
import random
import math
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle



class s2d3ds:
    def __init__(self, data_path='./raw', s3dis_path=None, cache_path=None, split_max_count=20, split_max_ratio=0.5, split_train_angle=(30, 'deg'), split_test_angle=(15, 'deg'), split_seed=0, verbose=True, debug=False, load_pointcloud=True, aligned=False, fixed=True):
        self.data_path = data_path
        self.s3dis_path = s3dis_path
        self.cache_path = cache_path
        
        self.split_max_count = split_max_count
        self.split_max_ratio = split_max_ratio
        self.split_train_angle = split_train_angle
        self.split_test_angle = split_test_angle
        self.split_seed = split_seed # None: completely random, but not reproducible
        self.verbose = verbose
        self.debug = debug
        self.load_pointcloud = load_pointcloud # False: caching might go wrong
        self.aligned = aligned
        self.fixed = fixed

        if isinstance(self.split_train_angle, list) or isinstance(self.split_train_angle, tuple):
            if self.split_train_angle[-1] == 'deg':
                self.split_train_angle = self.split_train_angle[0] * np.pi / 180.0
            else:
                self.split_train_angle = self.split_train_angle[0]
        if isinstance(self.split_test_angle, list) or isinstance(self.split_test_angle, tuple):
            if self.split_test_angle[-1] == 'deg':
                self.split_test_angle = self.split_test_angle[0] * np.pi / 180.0
            else:
                self.split_test_angle = self.split_test_angle[0]

        self.folders = ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6']
        self.folders = [os.path.join(self.data_path, folder) for folder in self.folders]
        self.folders = list(filter(lambda x: os.path.isfile(os.path.join(x, '3d/pointcloud.mat')), self.folders))
        self.images = {}
        self.poses = {}
        self.pc2exr = {}
        self.pointclouds = {}
        self.backprojected_pointclouds = {}

        self.train = {} # {area: [idx, ...]}
        self.test = {}
    
        self.load_files()
        self.split()
    

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def tqdm(self, x, desc='', leave=True):
        if self.verbose:
            return tqdm(x, desc=desc, leave=leave)
        else:
            return x


    def load_files(self):
        for folder in self.tqdm(self.folders, desc='area - image & loading cached', leave=True):
            folder_idx = folder.split('_')[-1][0] # '1' ~ '6'
            cache_path = None
            if self.cache_path is not None:
                if self.debug:
                    cache_path = os.path.join(self.cache_path, '%s_debug.pickle'%folder_idx)
                else:
                    cache_path = os.path.join(self.cache_path, '%s.pickle'%folder_idx)
                if os.path.isfile(cache_path):
                    if folder_idx not in self.images:
                        with open(cache_path, 'rb') as f:
                            self.images[folder_idx], self.poses[folder_idx], self.pointclouds[folder_idx] = pickle.load(f)
                    continue
            if folder_idx not in self.images:
                self.images[folder_idx] = []
                self.poses[folder_idx] = []
            pose_files = sorted(glob.glob(os.path.join(folder, 'data', 'pose', '*.json')))
            for pfn in self.tqdm(pose_files, desc='files', leave=False):
                ifn = pfn.replace('/data/pose/', '/data/rgb/').replace('pose.json', 'rgb.png')
                if os.path.isfile(ifn):
                    self.images[folder_idx].append(ifn)
                    self.poses[folder_idx].append(pfn)

        if self.load_pointcloud:
            import mat73
            if self.s3dis_path is None:
                for folder in self.tqdm(self.folders, desc='area - pointcloud', leave=True):
                    folder_idx = folder.split('_')[-1][0] # '1' ~ '6'
                    if folder_idx in self.pointclouds:
                        continue
                    self.pointclouds[folder_idx] = []
                    for _ in self.tqdm(range(1), desc='loading point cloud mat', leave=False):
                        places = mat73.loadmat(os.path.join(folder, '3d', 'pointcloud.mat'))['A' + folder.split('/')[-1][1:]]['Disjoint_Space']
                    for place in self.tqdm(places, desc='places', leave=False):
                        for part in place['object']['points']:
                            if self.debug:
                                part = part[::100]
                            # https://github.com/alexsax/2D-3D-Semantics/issues/40
                            if self.fixed and folder == 'area_5b':
                                part = part[:, [2, 1, 0]]
                                part[:, 0] = -part[:, 0]
                                bias = np.array([-4.09703582e+00, 3.27508322e-04, -6.22617759e+00])
                                part += bias
                            self.pointclouds[folder_idx].append(part)
            else:
                not_cached = []
                for k in self.images:
                    if k not in self.pointclouds:
                        not_cached.append(k)
                if len(not_cached) > 0:
                    for _ in self.tqdm(range(1), desc='loading point cloud mat', leave=False):
                        if not self.aligned:
                            pc_mat = mat73.loadmat(os.path.join(self.s3dis_path, 'Stanford3dDataset_v1.2.mat'))
                        else:
                            pc_mat = mat73.loadmat(os.path.join(self.s3dis_path, 'Stanford3dDataset_v1.2_Aligned_Version.mat'))
                        if 'Area' in pc_mat:
                            pc_mat = pc_mat['Area']
                        elif 'AlignedArea' in pc_mat:
                            pc_mat = pc_mat['AlignedArea']
                    for i in self.tqdm(range(len(pc_mat['name'])), desc='area - pointcloud', leave=True):
                        folder_idx = pc_mat['name'][i][-1]
                        if folder_idx not in not_cached:
                            continue
                        self.pointclouds[folder_idx] = []
                        place = pc_mat['Disjoint_Space'][i]
                        for object in place['object']:
                            for part in object['points']:
                                if self.debug:
                                    part = part[::100]
                                self.pointclouds[folder_idx].append(part)

        for k in self.pointclouds:
            if isinstance(self.pointclouds[k], list):
                self.pointclouds[k] = np.vstack(self.pointclouds[k]).astype(np.float64)
            if self.cache_path is not None:
                if self.debug:
                    cache_path = os.path.join(self.cache_path, '%s_debug.pickle'%k)
                else:
                    cache_path = os.path.join(self.cache_path, '%s.pickle'%k)
                if not os.path.isfile(cache_path):
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        data = self.images[k], self.poses[k], self.pointclouds[k]
                        pickle.dump(data, f)

    def split(self):
        cache_path = None
        if self.cache_path is not None and self.split_seed is not None:
            cache_path = os.path.join(self.cache_path, 'split_%d_%f_%f_%f_%d.pickle'%(self.split_max_count, self.split_max_ratio, self.split_train_angle, self.split_test_angle, self.split_seed))
            if os.path.isfile(cache_path):
                with open(cache_path, 'rb') as f:
                    self.train, self.test = pickle.load(f)
                for k in self.images:
                    self.print('area%s: %d images (%d train, %d test)'%(k, self.len_image(k), len(self.train[k]), len(self.test[k])))
                return
        for k in self.tqdm(self.images, desc='split area', leave=True):
            if self.split_seed is not None:
                random.seed(self.split_seed)
            self.train[k] = []
            self.test[k] = []
            set_dict = {}
            for idx, img_path in self.tqdm(enumerate(self.images[k]), desc='files', leave=False):
                camera, uuid, place1, place2, frame, frame_idx, domain, rgb = img_path.split('/')[-1].split('.')[0].split('_')
                assert camera == 'camera' and frame == 'frame' and domain == 'domain' and rgb == 'rgb'
                key = (uuid, place1, place2)
                if key not in set_dict:
                    set_dict[key] = []
                set_dict[key].append(idx)
            for key in self.tqdm(set_dict, desc='sampling', leave=False):
                test_count = min(self.split_max_count, math.floor(len(set_dict[key]) * self.split_max_ratio))
                random.shuffle(set_dict[key])
                if self.split_test_angle is None:
                    self.test[k] += set_dict[key][:test_count]
                    train_candidate = set_dict[key][test_count:]
                else:
                    train_candidate = []
                    angles = []
                    for idx in set_dict[key]:
                        pose, _ = self.get_pose(idx=idx, area=k)
                        angle = pose[2, :3]
                        available = True
                        for ang in angles:
                            if (angle * ang).sum() > np.cos(self.split_test_angle):
                                available = False
                                break
                        if available and test_count > 0:
                            angles.append(angle)
                            self.test[k].append(idx)
                            test_count -= 1
                        else:
                            train_candidate.append(idx)
                train_candidate = train_candidate[::-1]
                if self.split_train_angle is None:
                    self.train[k] = train_candidate
                else:
                    angles = []
                    for idx in train_candidate:
                        pose, _ = self.get_pose(idx=idx, area=k)
                        angle = pose[2, :3]
                        available = True
                        for ang in angles:
                            if (angle * ang).sum() > np.cos(self.split_train_angle):
                                available = False
                                break
                        if available:
                            angles.append(angle)
                            self.train[k].append(idx)
            self.train[k] = sorted(self.train[k])
            self.test[k] = sorted(self.test[k])
            self.print('area%s: %d images (%d train, %d test)'%(k, self.len_image(k), len(self.train[k]), len(self.test[k])))
        if cache_path is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump((self.train, self.test), f)
    def load_image(self, img_path, area='area_1'):
        if isinstance(img_path, str):
            return np.array(Image.open(img_path))
        else:
            area = area.split('_')[-1][0]
            return self.load_image(self.images[area][img_path])

    def load_pose(self, pose_path, area='area_1'):
        if isinstance(pose_path, str):
            with open(pose_path, 'r') as f:
                return json.load(f)
        else:
            area = area.split('_')[-1][0]
            return self.load_pose(self.poses[area][pose_path])
    
    def get_area(self):
        return sorted(list(self.images.keys()))

    def len_image(self, area='area_1'):
        area = area.split('_')[-1][0]
        return len(self.images[area])

    def get_pose(self, idx=0, area='area_1'):
        area = area.split('_')[-1][0]
        pose_path = self.poses[area][idx]
        pose_json = self.load_pose(pose_path)
        intrinsic = np.array(pose_json['camera_k_matrix'])
        pose = np.eye(4)
        pose[:3, 3] = np.array(pose_json['camera_location'])
        # A = self.euler2rot(pose_json['camera_original_rotation'])
        # pose[:3, :3] = A @ np.array(pose_json['camera_rt_matrix'])[:3, :3]) @ A
        pose[:3, :3] = self.euler2rot(pose_json['final_camera_rotation']) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        pose = np.linalg.inv(pose)
        return pose, intrinsic # world_to_cam camera pose, intrinsic

    def get_image(self, idx=0, area='area_1', load_image=True):
        area = area.split('_')[-1][0]
        img_path = self.images[area][idx]
        if load_image:
            image = self.load_image(img_path)
        else:
            image = img_path
        pose, intrinsic = self.get_pose(idx, area)
        return image, pose, intrinsic

    def get_depthimage(self, idx=0, area='area_1', load_image=True):
        area = area.split('_')[-1][0]
        img_path = self.images[area][idx].replace('/data/rgb/', '/data/depth/').replace('_rgb.png', '_depth.png')
        if load_image:
            image = self.load_image(img_path) / 512.0
        else:
            image = img_path
        pose, intrinsic = self.get_pose(idx, area)
        return image, pose, intrinsic

    def get_pointcloud(self, area='area_1'):
        area = area.split('_')[-1][0]
        return self.pointclouds[area]

    def get_backprojected_pointcloud(self, area='area_1', scale=1):
        area = area.split('_')[-1][0]
        cache_path = None
        if self.cache_path is not None:
            if self.debug:
                cache_path = os.path.join(self.cache_path, '%s_backprojected_debug.pickle'%area)
            else:
                cache_path = os.path.join(self.cache_path, '%s_backprojected.pickle'%area)
        if area in self.backprojected_pointclouds:
            return self.backprojected_pointclouds[area][::scale]
        elif cache_path is not None and os.path.isfile(cache_path):
            with open(cache_path, 'rb') as f:
                self.backprojected_pointclouds[area] = pickle.load(f)
            return self.backprojected_pointclouds[area][::scale]
        else:
            self.backprojected_pointclouds[area] = []
            pc_tqdm = tqdm(range(self.len_image(area)), desc='backprojected pointcloud', leave=True)
            for i in pc_tqdm:
                depth, pose, intrinsic = self.get_depthimage(idx=i, area=area)
                pose = np.linalg.inv(pose) # cam to world
                h, w = depth.shape[:2]
                xyz = np.zeros((h, w, 3))
                xyz[:, :, 0] = (np.arange(w).reshape((1, w)) - intrinsic[0][-1]) / intrinsic[0][0]
                xyz[:, :, 1] = (np.arange(h).reshape((h, 1)) - intrinsic[1][-1]) / intrinsic[1][1]
                xyz[:, :, 2] = 1.0
                depth_ind = (depth != 65535)
                xyz = depth[depth_ind].reshape((-1, 1)) * xyz[depth_ind].reshape((-1, 3))
                if self.debug:
                    xyz = xyz[::10000]
                xyz = xyz.T
                xyz = np.vstack([xyz, np.ones_like(xyz[:1])])
                xyz = (pose @ xyz).T
                self.backprojected_pointclouds[area].append(xyz)
            self.backprojected_pointclouds[area] = np.vstack(self.backprojected_pointclouds[area])
            if cache_path is not None:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.backprojected_pointclouds[area], f)
    
    def euler2rot(self, angle, reverse=False):
        phi = angle[0]
        theta = angle[1]
        psi = angle[2]
        ax = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]])
        ay = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]])
        az = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]])
        if reverse:
            return ax @ ay @ az
        return az @ ay @ ax
    
    def project_lidar(self, image, image_pose, intrinsic, area='area_1', max_z=10):
        if isinstance(image, str):
            image = self.load_image(image)
        area = area.split('_')[-1][0] # '1' ~ '6'

        lidar = np.hstack([self.pointclouds[area], np.ones_like(self.pointclouds[area][:, :1])]).T
        lidar_camera_coord = image_pose @ lidar
        lidar_camera_coord = lidar_camera_coord[:3, :] / lidar_camera_coord[3:, :]
        lidar_image_coord = intrinsic @ lidar_camera_coord
        lidar_camera_coord = lidar_camera_coord.T
        lidar_image_coord = lidar_image_coord.T

        # lidar in camera / image coordinate system (in front of camera)
        ind = np.logical_and(lidar_camera_coord[:, -1] > 0.0, lidar_camera_coord[:, -1] < max_z)
        lidar_camera_coord = lidar_camera_coord[ind]
        lidar_image_coord = lidar_image_coord[ind]
        lidar_image_coord = lidar_image_coord[:, :2] / lidar_image_coord[:, 2:] # (x, y, z) -> (x/z, y/z)

        # lidar in camera / image coordinate system (in front of camera)
        h, w = image.shape[:2]
        ind = np.all([lidar_image_coord[:, 0] >= 0.0, lidar_image_coord[:, 1] >= 0.0, lidar_image_coord[:, 0] <= w - 1, lidar_image_coord[:, 1] <= h - 1], axis=0)
        lidar_camera_coord = lidar_camera_coord[ind]
        lidar_image_coord = lidar_image_coord[ind]

        # draw on image
        xy = np.round(lidar_image_coord).astype(np.int32)
        image_project = image.copy()
        image_project[xy[:, 1], xy[:, 0]] = (image_project[xy[:, 1], xy[:, 0]]) // 2
        
        return image_project, lidar_camera_coord, xy
    
    def downsample(self, pointcloud, num_points=4096):
        import pcl
        MAX_STEP = 64
        pointcloud = pcl.PointCloud(pointcloud.astype(np.float32))
        if pointcloud.size <= num_points:
            if pointcloud.size < num_points:
                print('Downsampling failed. ' + str(pointcloud.shape[0]) + 'points...')
                return np.asarray(pointcloud)
            return np.asarray(pointcloud)
        sor = pointcloud.make_voxel_grid_filter()
        voxel_size = 0.0
        delta = 10.0
        while True:
            sor.set_leaf_size(delta, delta, delta)
            pc_small = sor.filter()
            if pc_small.size >= num_points:
                break
            delta *= 0.5

        for step in range(MAX_STEP):
            prev_voxel_size = voxel_size
            voxel_size += delta
            sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
            pc_small = sor.filter()
            if pc_small.size == num_points:
                break
            elif pc_small.size < num_points:
                voxel_size = prev_voxel_size
            delta *= 0.5

        sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
        pointcloud = sor.filter()

        pointcloud = np.asarray(pointcloud)
        if pointcloud.shape[0] > num_points:
            # print('Downsampling result: ' + str(pointcloud.size) + ' points, cutting...')
            ind = pointcloud[:, 2].argsort()[:num_points]
            pointcloud = pointcloud[ind]
        return pointcloud.astype(np.float64)



class KITTI:
    def __init__(self, dataset_path='../data', sequences=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], pose_mode='lidar0'):
        self.dataset_path = dataset_path
        self.sequences = sequences
        self.pose_mode = pose_mode # one of 'lidar0', 'camera0'. camera0 = world coordinate, lidar0 = 0th lidar coordinate
        
        self.poses = {} # world to lidar
        
        self.cam0_files = {}
        self.cam1_files = {}
        self.cam2_files = {}
        self.cam3_files = {}
        self.lidar_files = {}
        self.times = {}
        self.extrinsics = {} # lidar to sensor (cam only)
        self.intrinsics = {}

        self.load_files()
        self.load_poses()

    def load_files(self):
        name_dict = {'cam0': 'image_0', 'cam1': 'image_1', 'cam2': 'image_2', 'cam3': 'image_3', 'lidar': 'velodyne'}
        for s in self.sequences:
            seq_dir = os.path.join(self.dataset_path, 'sequences', s)
            for sensor, dir in name_dict.items():
                exec('self.%s_files[s] = sorted(glob.glob(os.path.join(seq_dir, dir, "*.*")))'%sensor)
            with open(os.path.join(seq_dir, 'times.txt'), 'r') as f:
                lines = f.readlines()
                self.times[s] = [float(line) for line in lines]
            calib_file = os.path.join(seq_dir, 'calib_tr.txt')
            if not os.path.isfile(calib_file):
                calib_file = os.path.join(seq_dir, 'calib.txt')
            with open(calib_file, 'r') as f:
                lines = f.readlines()
                self.extrinsics[s] = {}
                self.intrinsics[s] = {}
                e = {}
                for line in lines:
                    k, line = line.split(':')
                    e[k] = np.vstack([np.array([float(x) for x in line.split()]).reshape((3, 4)), np.array([[0.0, 0.0, 0.0, 1.0]])])
                    if k[0] == 'P':
                        self.intrinsics[s][k] = e[k][:3, :3]
                for k, v in e.items():
                    if k != 'Tr':
                        T = np.eye(4)
                        # T[0, 3] = v[0, 3] / v[0, 0]
                        T[0:3, 3] = v[0:3, 3] / v[0, 0]
                        self.extrinsics[s][k] = T @ e['Tr']

    def load_poses(self):
        for s in self.sequences:
            with open(os.path.join(self.dataset_path, 'poses', '%s.txt'%s), 'r') as f:
                lines = f.readlines()
                self.poses[s] = [(np.linalg.inv(self.extrinsics[s]['P0']) @ np.linalg.inv(np.vstack([np.array([float(x) for x in line.split()]).reshape((3, 4)), np.array([[0.0, 0.0, 0.0, 1.0]])]))) for line in lines]


    def len_seq(self, seq='00'):
        return len(self.times[seq])

    def get_lidar(self, seq='00', idx=0):
        if idx < 0 or idx >= self.len_seq(seq):
            return None, None, None
        lidar = self.lidar_files[seq][idx]
        time = self.times[seq][idx]
        pose = self.poses[seq][idx]
        if self.pose_mode == 'lidar0':
            pose = pose @ np.linalg.inv(self.poses[seq][0])
        return lidar, time, pose

    def get_image(self, seq='00', sensor=0, idx=0):
        if idx < 0 or idx >= self.len_seq(seq):
            return None, None, None, None
        if isinstance(sensor, str):
            sensor = int(sensor[-1])
        image = eval('self.cam%d_files[seq][idx]'%sensor)
        time = self.times[seq][idx]
        pose = self.extrinsics[seq]['P%d'%sensor] @ self.poses[seq][idx]
        if self.pose_mode == 'lidar0':
            pose = pose @ np.linalg.inv(self.poses[seq][0])
        K = self.intrinsics[seq]['P%d'%sensor]
        return image, time, pose, K
    
    def load_lidar(self, lidar_path, coord='xyz1'):
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        transpose = False
        if coord[-1] == 'T':
            coord = coord[:-1]
            transpose = True
        if coord == 'xyz':
            pass
        elif coord == 'xyz1':
            pc = np.hstack([pc, np.ones_like(pc[:, :1])])
        else:
            pass
        if transpose:
            pc = pc.T
        return pc
        
    def load_image(self, image_path):
        img = np.array(Image.open(image_path))
        return img
    
    def project_lidar(self, lidar, lidar_pose, image, image_pose, intrinsic):
        if isinstance(lidar, str):
            lidar = self.load_lidar(lidar)
        lidar = lidar.T
        if isinstance(image, str):
            image = self.load_image(image)
        
        # lidar in camera / image coordinate system (all points)
        lidar_camera_coord = image_pose @ np.linalg.inv(lidar_pose) @ lidar
        lidar_camera_coord = lidar_camera_coord[:3, :] / lidar_camera_coord[3:, :]
        lidar_image_coord = intrinsic @ lidar_camera_coord
        lidar_camera_coord = lidar_camera_coord.T
        lidar_image_coord = lidar_image_coord.T

        # lidar in camera / image coordinate system (in front of camera)
        ind = lidar_image_coord[:, -1] > 0.0
        lidar_camera_coord = lidar_camera_coord[ind]
        lidar_image_coord = lidar_image_coord[ind]
        lidar_image_coord = lidar_image_coord[:, :2] / lidar_image_coord[:, 2:] # (x, y, z) -> (x/z, y/z)

        # lidar in camera / image coordinate system (in front of camera)
        h, w = image.shape[:2]
        ind = np.all([lidar_image_coord[:, 0] >= 0.0, lidar_image_coord[:, 1] >= 0.0, lidar_image_coord[:, 0] <= w - 1, lidar_image_coord[:, 1] <= h - 1], axis=0)
        lidar_camera_coord = lidar_camera_coord[ind]
        lidar_image_coord = lidar_image_coord[ind]

        # draw on image
        xy = lidar_image_coord.astype(np.int32)
        image_project = image.copy()
        image_project[xy[:, 1], xy[:, 0]] = (image_project[xy[:, 1], xy[:, 0]]) // 2
        
        return image_project, lidar_camera_coord, xy
    
    def downsample(self, pointcloud, num_points=4096):
        import pcl
        MAX_STEP = 64
        pointcloud = pcl.PointCloud(pointcloud.astype(np.float32))
        if pointcloud.size <= num_points:
            if pointcloud.size < num_points:
                print('Downsampling failed. ' + str(pointcloud.shape[0]) + 'points...')
                return np.asarray(pointcloud)
            return np.asarray(pointcloud)
        sor = pointcloud.make_voxel_grid_filter()
        voxel_size = 0.0
        delta = 10.0
        while True:
            sor.set_leaf_size(delta, delta, delta)
            pc_small = sor.filter()
            if pc_small.size >= num_points:
                break
            delta *= 0.5

        for step in range(MAX_STEP):
            prev_voxel_size = voxel_size
            voxel_size += delta
            sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
            pc_small = sor.filter()
            if pc_small.size == num_points:
                break
            elif pc_small.size < num_points:
                voxel_size = prev_voxel_size
            delta *= 0.5

        sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
        pointcloud = sor.filter()

        pointcloud = np.asarray(pointcloud)
        if pointcloud.shape[0] > num_points:
            # print('Downsampling result: ' + str(pointcloud.size) + ' points, cutting...')
            ind = pointcloud[:, 2].argsort()[:num_points]
            pointcloud = pointcloud[ind]
        return pointcloud.astype(np.float64)
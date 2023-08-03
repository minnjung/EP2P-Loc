import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from multiprocessing import Process

from dataset_classes import s2d3ds



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/2d-3d-s', type=str, help='2D-3D-S dataset folder')
parser.add_argument('--s3dis_path', default='./data/s3dis', type=str, help='S3DIS dataset folder')
parser.add_argument('--cache_path', default='./data/cache', type=str, help='cache folder (optional)')
parser.add_argument('--save_path', default='./dataset/2d-3d-s', type=str, help='folder to save processed files')
parser.add_argument('--use_depth', default='5', type=str, help='list of areas to use depth map(RGB-D)')

# use default values to reproduce
parser.add_argument('--visible_distance', default=10.0, type=float, help='collect points within 0~visible_distance(m)')
parser.add_argument('--pc_points', default=65536, type=int, help='downsample to pc_points points')
parser.add_argument('--area_split', default='all', type=str, help='area split(fold) to preprocess. (ex. 12346_5, 1356_24, 245_136, all)')
parser.add_argument('--train_angle', default=30, type=int, help='angle threshold for train')
parser.add_argument('--test_angle', default=15, type=int, help='angle threshold for test')
parser.add_argument('--img_h', default=1080, type=int)
parser.add_argument('--img_w', default=1080, type=int)
parser.add_argument('--block_size', default=1.0, type=float, help='block size for fast checking visible points in the image frame')
parser.add_argument('--process', default=16, type=int, help='number of processes to run')
parser.add_argument('--exist_ok', default=True, type=bool, help='continue preprocessing if files exist')
args = parser.parse_args()

data_path = args.data_path
s3dis_path = args.s3dis_path
cache_path = args.cache_path
save_path = args.save_path
use_depth = args.use_depth

visible_distance = args.visible_distance
pc_points = args.pc_points
area_split_list = ['12346_5', '1356_24', '245_136'] # train_test
if args.area_split not in ['', 'all']:
    area_split_list = [args.area_split]

exist_ok = args.exist_ok


dataset = s2d3ds(data_path=data_path, s3dis_path=s3dis_path, debug=False, cache_path=cache_path, split_train_angle=(args.train_angle, 'deg'), split_test_angle=(args.test_angle, 'deg'))

# do not continue preprocessing
if not exist_ok:
    os.system('rm -rf %s'%save_path)

def build_image_and_pointcloud(id, process_num, global_pc, pc_lr, pc_k):
    idx_list = dataset.train[area] + dataset.test[area]
    n = len(idx_list)
    for idx in tqdm(idx_list[(id * n // process_num):((id + 1) * n // process_num)], desc='data: Process %02d / %02d'%(id, process_num), leave=False):
        image, image_pose, intrinsic = dataset.get_image(idx=idx, area=area, load_image=False)
        name = image.split('/')[-1].split('.')[0]
        img_path = os.path.join(save_path, area, 'image', '%s.png'%name)
        pc_path = os.path.join(save_path, area, 'pointcloud', '%s.npy'%name)
        if not os.path.isfile(img_path):
            os.system('ln -s %s %s'%(image, img_path))
        if not os.path.isfile(pc_path):
            if area not in use_depth:
                gpc = []
                lgpc = 0
                for i in range(len(pc_k)):
                    block_coord = image_pose @ pc_k[i]
                    block_coord = (block_coord[:3] / block_coord[3:]).reshape((3,))
                    if block_coord[2] > -np.sqrt(3) * args.block_size and block_coord[2] < visible_distance + np.sqrt(3) * args.block_size:
                        gpc.append(global_pc[:, pc_lr[i][0]:pc_lr[i][1]])
                lgpc = len(gpc)
                gpc = np.hstack(gpc)
                pc_cam = image_pose @ gpc
                pc_cam = pc_cam[:3] / pc_cam[3:]
                pc_img = intrinsic @ pc_cam
                pc_img = (pc_img[:2] / pc_img[2:])
                pc_img = np.round(pc_img)

                if args.img_h <= 0 or args.img_w <= 0:
                    h, w = np.array(Image.open(img_path)).shape[:2]
                else:
                    h, w = args.img_h, args.img_w
                conditions = []
                conditions += [pc_cam[2, :] > 0, pc_cam[2, :] < visible_distance]
                conditions += [pc_img[0, :] >= 0, pc_img[0, :] <= w - 1]
                conditions += [pc_img[1, :] >= 0, pc_img[1, :] <= h - 1]
                pc = gpc[:, np.all(np.array(conditions), axis=0)][:3, :].T
                if pc.shape[0] < pc_points:
                    area_tqdm.write('Warning [smaller point cloud %d for idx %d]: %s'%(pc.shape[0], idx, pc_path))
                    if pc.shape[0] == 0:
                        area_tqdm.write('Error [0points, pc_lr: %d, pc_k: %d]'%(len(pc_lr), len(pc_k)))
                        area_tqdm.write(str(image_pose))
                        area_tqdm.write(str(lgpc))
                        area_tqdm.write(str(gpc.shape))
                        area_tqdm.write('%f %f'%(pc_cam[2,:].min(), pc_cam[2,:].max()))
                        np.save('error.npy', gpc[:3, :].T)
                    pc = np.vstack([pc, pc[np.random.randint(0, pc.shape[0], size=(pc_points - pc.shape[0],))]])
                pc = dataset.downsample(pc, num_points=pc_points)
                np.save(pc_path, pc)
            else:
                depth, depth_pose, intrinsic = dataset.get_depthimage(idx=idx, area=area)
                h, w = depth.shape[:2]
                xyz = np.zeros((h, w, 3))
                xyz[:, :, 0] = (np.arange(w).reshape((1, w)) - intrinsic[0][-1]) / intrinsic[0][0]
                xyz[:, :, 1] = (np.arange(h).reshape((h, 1)) - intrinsic[1][-1]) / intrinsic[1][1]
                xyz[:, :, 2] = 1.0
                depth_ind = depth < visible_distance # (depth != 65535 / 512.0)
                xyz = depth[depth_ind].reshape((-1, 1)) * xyz[depth_ind].reshape((-1, 3))
                xyz = xyz.T
                xyz = np.vstack([xyz, np.ones_like(xyz[:1])])
                xyz = (np.linalg.inv(depth_pose) @ xyz).T
                xyz = xyz[:, :3] / xyz[:, 3:]
                pc = dataset.downsample(xyz, num_points=pc_points)
                np.save(pc_path, pc)

os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'files'), exist_ok=True)
area_tqdm = tqdm(dataset.get_area(), desc='area')
for area in area_tqdm:
    os.makedirs(os.path.join(save_path, area), exist_ok=True)
    os.makedirs(os.path.join(save_path, area, 'image'), exist_ok=True)
    os.makedirs(os.path.join(save_path, area, 'pointcloud'), exist_ok=True)
    if area not in use_depth:
        global_pc = dataset.get_pointcloud(area) # N, 3
        area_tqdm.write('building blocks')
        global_idx = np.floor(global_pc / args.block_size).astype(np.int32)
        gi_for_sort = np.array([tuple(x) for x in global_idx], dtype=[('x', np.int32), ('y', np.int32), ('z', np.int32)])
        area_tqdm.write('sorting blocks')
        sorted_ind = np.argsort(gi_for_sort, order=('x', 'y', 'z'), axis=0)
        global_pc = global_pc[sorted_ind]
        global_idx = global_idx[sorted_ind]
        area_tqdm.write('splitting blocks')
        unique_values, unique_counts = np.unique(global_idx, axis=0, return_counts=True)
        start, end = 0, 0
        pc_lr = []
        pc_k = []
        for i in tqdm(range(unique_values.shape[0]), desc='parsing', leave=False):
            k = unique_values[i]
            end = start + unique_counts[i]
            pc_lr.append((start, end))
            assert np.all(global_idx[start:end].max(0) == k) and np.all(global_idx[start:end].min(0) == k)
            start = end
            pc_k.append(np.array(list(k.reshape((3,)) * args.block_size) + [1]).reshape((4, 1)))

        global_pc = np.hstack([global_pc, np.ones_like(global_pc[:, :1])]).T # 4, N
    else:
        global_pc, pc_lr, pc_k = None, None, None

    if args.process <= 1:
        build_image_and_pointcloud(0, 1, global_pc, pc_lr, pc_k)
    else:
        th = []
        for i in range(args.process):
            th.append(Process(target=build_image_and_pointcloud, args=(i, args.process, global_pc, pc_lr, pc_k)))
        for i in range(args.process):
            th[i].start()
        for i in range(args.process):
            th[i].join()

for area_split in tqdm(area_split_list, desc='area_split'):
    os.makedirs(os.path.join(save_path, 'files_' + area_split + '_%d_%d'%(args.train_angle, args.test_angle)), exist_ok=True)
    for area in tqdm(dataset.get_area(), desc='area', leave=False):
        with open(os.path.join(save_path, 'files_' + area_split + '_%d_%d'%(args.train_angle, args.test_angle), area + '_train.txt'), 'w') as f:
            for idx in tqdm(dataset.train[area], desc='data', leave=False):
                image, image_pose, intrinsic = dataset.get_image(idx=idx, area=area, load_image=False)
                name = image.split('/')[-1].split('.')[0]
                img_path = os.path.join(save_path, area, 'image', '%s.png'%name)
                pc_path = os.path.join(save_path, area, 'pointcloud', '%s.npy'%name)
                text = '%s %s'%(img_path, pc_path)
                for p in np.concatenate([image_pose.reshape((16,)), intrinsic.reshape((9,))], axis=0):
                    text += ' %f'%p
                f.write(text + '\n')
        with open(os.path.join(save_path, 'files_' + area_split + '_%d_%d'%(args.train_angle, args.test_angle), area + '_test.txt'), 'w') as f:
            for idx in tqdm(dataset.test[area], desc='data', leave=False):
                image, image_pose, intrinsic = dataset.get_image(idx=idx, area=area, load_image=False)
                name = image.split('/')[-1].split('.')[0]
                img_path = os.path.join(save_path, area, 'image', '%s.png'%name)
                pc_path = os.path.join(save_path, area, 'pointcloud', '%s.npy'%name)
                text = '%s %s'%(img_path, pc_path)
                for p in np.concatenate([image_pose.reshape((16,)), intrinsic.reshape((9,))], axis=0):
                    text += ' %f'%p
                f.write(text + '\n')

os.makedirs(os.path.join(save_path, 'files_%d_%d'%(args.train_angle, args.test_angle)), exist_ok=True)
for area in tqdm(dataset.get_area(), desc='area', leave=False):
    with open(os.path.join(save_path, 'files_%d_%d'%(args.train_angle, args.test_angle), area + '_train.txt'), 'w') as f:
        for idx in tqdm(dataset.train[area], desc='data', leave=False):
            image, image_pose, intrinsic = dataset.get_image(idx=idx, area=area, load_image=False)
            name = image.split('/')[-1].split('.')[0]
            img_path = os.path.join(save_path, area, 'image', '%s.png'%name)
            pc_path = os.path.join(save_path, area, 'pointcloud', '%s.npy'%name)
            text = '%s %s'%(img_path, pc_path)
            for p in np.concatenate([image_pose.reshape((16,)), intrinsic.reshape((9,))], axis=0):
                text += ' %f'%p
            f.write(text + '\n')
    with open(os.path.join(save_path, 'files_%d_%d'%(args.train_angle, args.test_angle), area + '_test.txt'), 'w') as f:
        for idx in tqdm(dataset.test[area], desc='data', leave=False):
            image, image_pose, intrinsic = dataset.get_image(idx=idx, area=area, load_image=False)
            name = image.split('/')[-1].split('.')[0]
            img_path = os.path.join(save_path, area, 'image', '%s.png'%name)
            pc_path = os.path.join(save_path, area, 'pointcloud', '%s.npy'%name)
            text = '%s %s'%(img_path, pc_path)
            for p in np.concatenate([image_pose.reshape((16,)), intrinsic.reshape((9,))], axis=0):
                text += ' %f'%p
            f.write(text + '\n')
import os

import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from multiprocessing import Process

from dataset_classes import KITTI



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/kitti', type=str, help='KITTI dataset folder')
parser.add_argument('--save_path', default='./dataset/kitti', type=str, help='folder to save processed files')

# use default values to reproduce
parser.add_argument('--visible_distance', default=30.0, type=float, help='collect points within 0~visible_distance(m)')
parser.add_argument('--pc_points', default=65536, type=int, help='downsample to pc_points points')
parser.add_argument('--img_h', default=0, type=int)
parser.add_argument('--img_w', default=0, type=int)
parser.add_argument('--pc_collect_time', default=60.0, type=float, help='only collect points within pc_collect_time (sec)')
parser.add_argument('--img_collect_distance', default=2.0, type=float, help='collect image every img_collect_distance(m)')
parser.add_argument('--camera_idx', default=2, type=int, help='usinng camera_idx th camera (0, 1: gray, 2, 3: color)')
parser.add_argument('--block_size', default=5.0, type=float, help='block size for fast checking visible points in the image frame')
parser.add_argument('--process', default=16, type=int, help='number of processes to run')
parser.add_argument('--exist_ok', default=True, type=bool, help='continue preprocessing if files exist')
args = parser.parse_args()

visible_distance = args.visible_distance
pc_points = args.pc_points
pc_collect_time = args.pc_collect_time
img_collect_distance = args.img_collect_distance
camera_idx = args.camera_idx

exist_ok = args.exist_ok
data_path = os.path.abspath(args.data_path)
save_path = os.path.abspath(args.save_path)


dataset = KITTI(dataset_path=data_path, pose_mode='lidar0')
if not exist_ok:
    os.system('rm -rf %s'%save_path)

def build_image_and_pointcloud(id, process_num, global_pc, global_t, pc_lr, pc_k, dataset_img):
    n = len(dataset_img)
    for idx in tqdm(range((id * n // process_num), ((id + 1) * n // process_num)), desc='data: Process %02d / %02d'%(id, process_num), leave=False):
        source_path, img_path, timestamp, image_pose, intrinsic = dataset_img[idx]
        name = source_path.split('/')[-1].split('.')[0]
        pc_path = os.path.join(save_path, seq, 'pointcloud', '%s.npy'%name)
        if not os.path.isfile(img_path):
            os.system('ln -s %s %s'%(source_path, img_path))
        if not os.path.isfile(pc_path):
            gpc = []
            for i in range(len(global_t)):
                if abs(global_t[i] - timestamp) < args.pc_collect_time:
                    for j in range(len(pc_k[i])):
                        block_coord = image_pose @ pc_k[i][j]
                        block_coord = (block_coord[:3] / block_coord[3:]).reshape((3,))
                        if block_coord[2] > -np.sqrt(3) * args.block_size and block_coord[2] < visible_distance + np.sqrt(3) * args.block_size:
                            gpc.append(global_pc[i][:, pc_lr[i][j][0]:pc_lr[i][j][1]])
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
            # conditions += [np.abs(gt - timestamp) < args.pc_collect_time]
            pc = gpc[:, np.all(np.array(conditions), axis=0)][:3, :].T
            if pc.shape[0] < pc_points:
                seq_tqdm.write('Warning [smaller point cloud %d]: %s'%(pc.shape[0], pc_path))
                pc = np.vstack([pc, pc[np.random.randint(0, pc.shape[0], size=(pc_points - pc.shape[0],))]])
            pc = dataset.downsample(pc, num_points=pc_points)
            np.save(pc_path, pc)

os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'files'), exist_ok=True)
seq_tqdm = tqdm(dataset.sequences, desc='sequences')
for seq in seq_tqdm:
    os.makedirs(os.path.join(save_path, seq), exist_ok=True)
    os.makedirs(os.path.join(save_path, seq, 'image'), exist_ok=True)
    os.makedirs(os.path.join(save_path, seq, 'pointcloud'), exist_ok=True)

    lidars = []
    timestamps = []

    dataset_image = []
    last_image_pose = None

    if os.path.isfile(os.path.join(save_path, '%s_img.pickle'%seq)):
        with open(os.path.join(save_path, '%s_img.pickle'%seq), 'rb') as f:
            lidars, timestamps, dataset_image = pickle.load(f)
    else:
        data_tqdm = tqdm(range(dataset.len_seq(seq)), desc='sequence' + seq, leave=False)
        for i in data_tqdm:
            lidar_path, timestamp, lidar_pose = dataset.get_lidar(seq=seq, idx=i)
            image_path, _, image_pose, intrinsic = dataset.get_image(seq=seq, idx=i, sensor=camera_idx)

            # point cloud
            lidar = dataset.load_lidar(lidar_path, coord='xyz1T') # 4, N
            lidar = np.linalg.inv(lidar_pose) @ lidar
            lidar = (lidar.T)[:, :3] # N, 3
            # timestamp = np.full((lidar.shape[0],), timestamp)
            lidars.append(lidar)
            timestamps.append(timestamp)

            # image
            p = np.linalg.inv(image_pose)[:3, 3]
            if last_image_pose is None or np.linalg.norm(last_image_pose - p, ord=2, axis=-1) >= img_collect_distance:
                image = dataset.load_image(image_path)
                img_path = os.path.join(save_path, seq, 'image', '%s.png'%image_path.split('/')[-1].split('.')[0])
                dataset_image.append([image_path, img_path, timestamp, image_pose, intrinsic])
                last_image_pose = p
        with open(os.path.join(save_path, '%s_img.pickle'%seq), 'wb') as f:
            pickle.dump((lidars, timestamps, dataset_image), f)

    pc_lr = []
    pc_k = []
    global_pc = []
    global_t = timestamps
    if os.path.isfile(os.path.join(save_path, '%s_pc.pickle'%seq)):
        with open(os.path.join(save_path, '%s_pc.pickle'%seq), 'rb') as f:
            global_pc, pc_lr, pc_k = pickle.load(f)
    else:
        for idx in tqdm(range(len(lidars)), desc='parse', leave=False):
            pc_lr.append([])
            pc_k.append([])
            global_idx = np.floor(lidars[idx] / args.block_size).astype(np.int32)
            gi_for_sort = np.array([tuple(x) for x in global_idx], dtype=[('x', np.int32), ('y', np.int32), ('z', np.int32)])
            sorted_ind = np.argsort(gi_for_sort, order=('x', 'y', 'z'), axis=0)
            lidars[idx] = lidars[idx][sorted_ind]
            global_idx = global_idx[sorted_ind]
            unique_values, unique_counts = np.unique(global_idx, axis=0, return_counts=True)
            start, end = 0, 0
            for i in range(unique_values.shape[0]):
                k = unique_values[i]
                end = start + unique_counts[i]
                pc_lr[idx].append((start, end))
                # assert np.all(global_idx[start:end].max(0) == k) and np.all(global_idx[start:end].min(0) == k)
                start = end
                pc_k[idx].append(np.array(list(k.reshape((3,)) * args.block_size) + [1]).reshape((4, 1)))
            lidars[idx] = np.hstack([lidars[idx], np.ones_like(lidars[idx][:, :1])]).T
        global_pc = lidars
        with open(os.path.join(save_path, '%s_pc.pickle'%seq), 'wb') as f:
            pickle.dump((global_pc, pc_lr, pc_k), f)

    if args.process <= 1:
        build_image_and_pointcloud(0, 1, global_pc, global_t, pc_lr, pc_k, dataset_image)
    else:
        th = []
        for i in range(args.process):
            th.append(Process(target=build_image_and_pointcloud, args=(i, args.process, global_pc, global_t, pc_lr, pc_k, dataset_image)))
        for i in range(args.process):
            th[i].start()
        for i in range(args.process):
            th[i].join()

    if seq in dataset.sequences[-2:]:
        split = 'test'
    else:
        split = 'train'
    with open(os.path.join(save_path, 'files', seq + '_%s.txt'%split), 'w') as f:
        for idx in tqdm(range(len(dataset_image)), desc='data', leave=False):
            _, image, timestamp, image_pose, intrinsic = dataset_image[idx]
            name = image.split('/')[-1].split('.')[0]
            pc_path = os.path.join(save_path, seq, 'pointcloud', '%s.npy'%name)
            text = '%s %s'%(image, pc_path)
            for p in np.concatenate([image_pose.reshape((16,)), intrinsic.reshape((9,))], axis=0):
                text += ' %f'%p
            f.write(text + '\n')
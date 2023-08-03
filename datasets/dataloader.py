import random
import torch
from torch.utils.data import DataLoader



def collate_fn(batch):
    data = None
    not_none_idx = -1
    for i in range(len(batch)):
        if batch[i] is not None:
            not_none_idx = i
            break
    if not_none_idx < 0:
        return [None for i in range(len(batch))]
    if isinstance(batch[not_none_idx], dict):
        data = {}
        for k in batch[not_none_idx]:
            data[k] = collate_fn([x[k] for x in batch])
    elif isinstance(batch[not_none_idx], list):
        data = []
        for i in range(len(batch[not_none_idx])):
            data.append(collate_fn([x[i] for x in batch]))
    elif isinstance(batch[not_none_idx], tuple):
        data = []
        for i in range(len(batch[not_none_idx])):
            data.append(collate_fn([x[i] for x in batch]))
        data = tuple(data)
    elif isinstance(batch[not_none_idx], torch.Tensor):
        data = torch.stack(batch, dim=0)
    elif isinstance(batch[not_none_idx], str):
        data = [x for x in batch]
    else:
        print('Not implemented for type %s'%str(type(batch[not_none_idx])))
        raise NotImplementedError
    return data


class BatchedNegativeLoader(DataLoader):
    def __init__(self, dataset, batch_size: int, shuffle: bool=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.negatives = self.dataset.num_negatives # copy from dataset
        self.dataset.num_negatives = 0
        self.data_keys = ['image', 'image_pose_quat', 'img_intrinsic', 'positives', 'positive_coords', 'depth_map', 'visible_depth_map']
        n_area = len(self.dataset.seq)
        assert n_area > 1 and self.batch_size >= self.negatives + ((self.negatives + n_area - 2) // (n_area - 1))

        self.batch_idx = []
        random.seed(0)

    def __iter__(self):
        if len(self.batch_idx) == 0:
            self.generate_batches()
        for batch in self.batch_idx:
            yield self.get_idx(batch)
        self.batch_idx = []

    def __len__(self):
        return len(self.batch_idx)

    def generate_batches(self):
        self.batch_idx = []

        unused_elements = list(range(len(self.dataset)))

        for retry_count in range(10):
            current_batch = []
            area_count = {k: 0 for k in range(len(self.dataset.seq))}
            used_elements = []
            if self.shuffle:
                random.shuffle(unused_elements)
            for idx in unused_elements:
                area = self.dataset.database_seq[idx].item()
                if area_count[area] + 1 + self.negatives > self.batch_size:
                    continue
                area_count[area] += 1
                current_batch.append(idx)
                if len(current_batch) >= self.batch_size:
                    self.batch_idx.append(current_batch)
                    used_elements += current_batch
                    current_batch = []
                    for k in area_count:
                        area_count[k] = 0
            unused_elements = list(set(unused_elements) - set(used_elements))
    
    def get_idx(self, batch):
        data = []
        areas = [self.dataset.database_seq[idx].item() for idx in batch]
        negatives = []
        for i, idx in enumerate(batch):
            d = self.dataset[idx]
            nd = {}
            for k in d:
                if k in self.data_keys:
                    nd[k] = d[k]
            data.append(nd)
            neg = []
            for j in range(len(batch)):
                if areas[i] != areas[j]:
                    neg.append(j)
            random.shuffle(neg)
            negatives.append(neg[:self.negatives])
        data = collate_fn(data)
        data['negatives'] = torch.tensor(negatives) # B, self.negatives
        return data


def get_dataloader(dataset, dataloader=None, batch_size=1, shuffle=True, drop_last=False, 
                   pin_memory=False, num_workers=8, persistent_workers=True):

    if isinstance(dataloader, str) and dataloader.lower() == 'batchednegativeloader':
        dataloader = BatchedNegativeLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

    return dataloader


def rgba_to_rgb(img):
    rgb_img = img[..., :3] * (img[..., -1:] / 255.0) + (1.0 - img[..., -1:] / 255.0)
    return rgb_img.astype(img.dtype)
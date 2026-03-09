"""DataLoader wrapper for training.

Original: src/data_io/dataset_loader.py (Minivision, zhuying)
"""
from __future__ import annotations

from torch.utils.data import DataLoader

from app.ml.data.dataset import DatasetFolderFT
from app.ml.data import transforms as trans


def get_train_loader(conf) -> DataLoader:
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size), scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
    ])
    root_path = f'{conf.train_root_path}/{conf.patch_info}'
    trainset = DatasetFolderFT(
        root_path, train_transform, None, conf.ft_width, conf.ft_height,
    )
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=conf.num_workers if hasattr(conf, 'num_workers') else 16,
    )
    return train_loader

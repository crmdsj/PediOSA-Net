import os
import pandas as pd
import numpy as np
import torch
from monai.data import PersistentDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Resized, Spacingd,
    Orientationd, ToTensorD, RandRotate90d, RandFlipd, RandAffined
)
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from monai.config import KeysCollection


def load_data(images_dir, label_path):
    df = pd.read_csv(label_path)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    image_paths = []
    labels = []
    no = []
    
    for index, row in df.iterrows():
        # 修改为 .npy 文件名
        sample_id = str(int(row['no']))
        image_name = f"{int(row['no'])}.npy"  # 假设文件名与 CSV 中的 'no' 列对应
        image_path = os.path.join(images_dir, image_name)
        
        if os.path.exists(image_path):
            no.append(sample_id)
            image_paths.append(image_path)
            labels.append(row['label'])
        else:
            print(f"Image not found: {image_path}")
    
    labels = np.array(labels, dtype=np.float32)
    return image_paths, labels,no


class LoadNpyDictd(LoadImaged):
    def __init__(self, keys: KeysCollection, **kwargs):
        super().__init__(keys, reader="NumpyReader", **kwargs)  # 指定使用 NumPy 读取器

def get_train_transform():
    """
    返回用于训练的转换流程，包括预处理和数据增强。
    """
    # 定义预处理流程
    pre_transforms = Compose([
        LoadNpyDictd(keys=["image"]),  # 加载 .npy 文件
        EnsureChannelFirstd(keys=["image"]),  # 添加通道维度 (C, H, W, D)
        NormalizeIntensityd(keys=["image"], nonzero=False, dtype=np.float32),
        Resized(keys=["image"], spatial_size=(64, 64, 64)),  # 调整尺寸
        ToTensorD(keys=["image", "label"]),
    ])
    
    # 定义数据增强流程（仅在训练时应用）
    aug_transforms = Compose([
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(1, 2)),  # 在横断面 (H, W) 旋转
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),           # 沿高度方向翻转
        RandAffined(
            keys=["image"],
            prob=0.5,
            rotate_range=(0, np.pi / 12),
            translate_range=(5, 5, 0),  # 在 H/W 方向平移
            scale_range=(0.1, 0.1, 0),  # 在 H/W 方向缩放
            mode='bilinear',
            padding_mode='border'
        ),
        Resized(keys=["image"], spatial_size=(64, 64, 64))  # 恢复尺寸
    ])
    
    # 合并预处理和数据增强
    return Compose([pre_transforms, aug_transforms])

def get_test_transform():
    """
    返回用于测试的转换流程，仅包括预处理。
    """
    # 定义预处理流程
    return Compose([
        LoadNpyDictd(keys=["image"]),  # 加载 .npy 文件
        EnsureChannelFirstd(keys=["image"]),  # 添加通道维度 (C, H, W, D)
        NormalizeIntensityd(keys=["image"], nonzero=False, dtype=np.float32),
        Resized(keys=["image"], spatial_size=(64, 64, 64)),  # 调整尺寸
        ToTensorD(keys=["image", "label"]),
    ])

def create_dataloader(images_dir: str,
                      label_path: str,
                      batch_size: int = 4,
                      num_workers: int = 0,
                      mode: str = "train",
                      cache_dir: str = "./cache_dir"):
    """
    images_dir: 存 npy 文件的文件夹
    label_path:  CSV 文件路径，含 no,label 列
    mode: 'train' or 'test'
    """
    image_paths, labels, ids = load_data(images_dir, label_path)
    if mode == "train":
        transform = get_train_transform()
        shuffle = True
    elif mode == "test":
        transform = get_test_transform()
        shuffle = False
    else:
        raise ValueError("mode must be 'train' or 'test'")

    data_dicts = [
        {"id": i, "image": img, "label": float(lbl)}
        for i, (img, lbl) in enumerate(zip(image_paths, labels))
    ]

    os.makedirs(cache_dir, exist_ok=True)
    ds = PersistentDataset(data=data_dicts,
                           transform=transform,
                           cache_dir=cache_dir)
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)
    return loader
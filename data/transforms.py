"""Data augmentation and helper utilities."""
import random
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, RandFlipd, RandAffined, RandAdjustContrastd,
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
    RandGaussianSmoothd, RandCoarseDropoutd, EnsureTyped
)


def get_monai_augment(with_coarse: bool):
    aug = [
        RandFlipd("image", prob=0.5, spatial_axis=[0]),
        RandFlipd("image", prob=0.5, spatial_axis=[1]),
        RandFlipd("image", prob=0.5, spatial_axis=[2]),
        RandAffined("image", prob=0.7,
                    rotate_range=(torch.pi/36,)*3,
                    translate_range=(4,4,4),
                    scale_range=(0.05,)*3,
                    padding_mode="border"),
        RandAdjustContrastd("image", prob=0.3, gamma=(0.7,1.4)),
        RandScaleIntensityd("image", factors=0.15, prob=0.3),
        RandShiftIntensityd("image", offsets=0.10, prob=0.3),
        RandGaussianNoised("image", prob=0.25, mean=0, std=0.01),
        RandGaussianSmoothd("image", prob=0.15,
                            sigma_x=(0,1), sigma_y=(0,1), sigma_z=(0,1)),
    ]
    if with_coarse:
        aug.append(RandCoarseDropoutd("image", prob=0.2,
                                      holes=4, spatial_size=(16,16,16),
                                      max_holes=4, fill_value=0))
    aug.append(EnsureTyped("image"))
    return Compose(aug)


def build_transforms(mean, std):
    load_tf = Compose([
        LoadImaged("image", image_only=False, reader="ITKReader"),
        EnsureChannelFirstd("image", strict_check=False),
        Orientationd("image", axcodes="SAR"),
        NormalizeIntensityd("image", subtrahend=mean, divisor=std),
        EnsureTyped("image")
    ])
    return load_tf, get_monai_augment(False), get_monai_augment(True)


def random_rotation_single(vol: torch.Tensor):
    axis = random.randint(0,2); k = random.randint(0,3)
    dims_map = {0:(2,3), 1:(1,3), 2:(1,2)}
    return torch.rot90(vol,k,dims=dims_map[axis]), axis*4+k

def random_jigsaw_single(vol: torch.Tensor):
    if random.random()<0.5:
        perm = torch.randperm(vol.shape[1]); return vol[:,perm],1
    return vol,0

def augment_batch(imgs: torch.Tensor, tf):
    return torch.stack([tf({"image": s})["image"] for s in imgs])

def create_mask(shape, ratio, device):
    return torch.rand(shape, device=device) < ratio

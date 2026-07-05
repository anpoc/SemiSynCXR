"""
PyTorch Dataset classes for CXR inpainting and evaluation.

Provides :class:`torch.utils.data.Dataset` wrappers that load row-indexed job dictionaries
(produced by :func:`inpaintGen.create_data_dict`) into batched tensors for diffusion
inpainting, similarity metrics, classification, CLIP scoring, and segmentation evaluation.

Dataset classes:
    - :class:`CXRDataset` — image + mask + prompt for inpainting (:mod:`inpaintGen`).
    - :class:`SimDataset` — generated vs. base image pairs for perceptual similarity.
    - :class:`EvalDataset` — single images with labels for classification evaluation.
    - :class:`CLIPDataset` — RGB images with prompts for CLIP-score evaluation.
    - :class:`SegDataset` — images with optional dual transforms for segmentation models.

Consumers:
    - :mod:`inpaintGen` — :class:`CXRDataset`.
    - :mod:`inpaintEval` — :class:`EvalDataset`, :class:`CLIPDataset`, :class:`SimDataset`,
      :class:`SegDataset`.
"""

import os
import torch
import re
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class CXRDataset(Dataset):
    """PyTorch dataset for inpainting jobs (image, mask, label, prompt).

    Args:
        data_dict (dict): Row-indexed job dict with keys ``img_file``, ``mask_id``,
            ``label``, ``prompt``.
        img_dir (str): Root directory for source CXR images.
        mask_dir (str): Root directory for mask PNGs (``{mask_id}.png``).
        transform (callable, optional): Torchvision transform applied to both image and mask.
    """

    def __init__(self, data_dict, img_dir, mask_dir, transform=None):
        self.data_dict = data_dict
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """Load one inpainting job as a dict of tensors and metadata strings."""
        img_path = os.path.join(self.img_dir, self.data_dict[idx]['img_file'])
        img = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.mask_dir, f"{self.data_dict[idx]['mask_id']}.png")
        mask = Image.open(mask_path)
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        sample = {'img': img, 'mask': mask, 'label': self.data_dict[idx]['label'], 'prompt': self.data_dict[idx]['prompt'], 
            'ids': [self.data_dict[idx]['img_file'], self.data_dict[idx]['mask_id']]}
        return sample


class SimDataset(Dataset):
    """Dataset of generated vs. base image pairs for perceptual similarity metrics.

    Args:
        data_dict (dict): Row-indexed dict with ``img_gen_file`` and ``img_base_file``.
        img_gen_dir (str): Directory of generated (inpainted) images.
        img_base_dir (str): Directory of original source images.
        transform_gen (callable, optional): Transform for generated images.
        transform_base (callable, optional): Transform for base images.
        transform_torch (bool): If True, apply transforms to torch tensors. Defaults to True.
    """

    def __init__(self, data_dict, img_gen_dir, img_base_dir, transform_gen=None, transform_base=None,
        transform_torch:bool=True):
        self.data_dict = data_dict
        self.img_gen_dir = img_gen_dir
        self.img_base_dir = img_base_dir
        self.transform_gen = transform_gen
        self.transform_base = transform_base
        self.transform_torch = transform_torch

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """Return scaled generated and base image tensors."""
        img_gen = get_image_scaled(
            img_path=os.path.join(self.img_gen_dir, f"{self.data_dict[idx]['img_gen_file']}"), 
            transform=self.transform_gen, transform_torch=self.transform_torch
        )
        img_base = get_image_scaled(
            img_path=os.path.join(self.img_base_dir, f"{self.data_dict[idx]['img_base_file']}"), 
            transform=self.transform_base, transform_torch=self.transform_torch
        )
        sample = {'img_gen': img_gen, 'img_base': img_base, 'img_gen_id': re.sub(r'(\.png)|(\.jpg)', '', 
            self.data_dict[idx]['img_gen_file'])}
        return sample


class EvalDataset(Dataset):
    """Dataset for classifier-based evaluation on single images.

    Args:
        data_dict (dict): Row-indexed dict with ``img_file`` and ``label``.
        img_dir (str): Directory containing evaluation images.
        transform (callable, optional): Pre-scaling transform.
        transform_torch (bool): If True, apply transform to torch tensor. Defaults to False.
    """

    def __init__(self, data_dict, img_dir, transform=None, transform_torch:bool=False):
        self.data_dict = data_dict
        self.img_dir = img_dir
        self.transform = transform
        self.transform_torch = transform_torch

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """Return a grayscale-scaled image tensor and label."""
        img = get_image_scaled(
            img_path=os.path.join(self.img_dir, f"{self.data_dict[idx]['img_file']}"), transform=self.transform, 
            transform_torch=self.transform_torch, convert_to='L', scale=1024
        )
        sample = {'img': img, 'label': self.data_dict[idx]['label'], 'img_id': re.sub(r'(\.png)|(\.jpg)', '', 
            self.data_dict[idx]['img_file'])}
        return sample


class CLIPDataset(EvalDataset):
    """Dataset for CXR-CLIP image–text similarity evaluation.

    Loads RGB images as ``(C, H, W)`` torch tensors without :func:`feature_scaling`.
    """

    def __getitem__(self, idx):
        """Return RGB image tensor, label, prompt, and image ID."""
        img = Image.open(os.path.join(self.img_dir, f"{self.data_dict[idx]['img_file']}")).convert('RGB')
        img = torch.tensor(list(img.getdata())).reshape(img.size[1], img.size[0], 3).permute(2, 0, 1)
        sample = {'img': img, 'label': self.data_dict[idx]['label'], 'prompt': self.data_dict[idx]['prompt'],
            'img_id': re.sub(r'(\.png)|(\.jpg)', '',  self.data_dict[idx]['img_file'])}
        return sample


class SegDataset(Dataset):
    """Dataset for segmentation-model evaluation with optional dual transforms.

    Args:
        data_dict (dict): Row-indexed dict with ``img_file``.
        img_dir (str): Directory containing images.
        transform (callable or list, optional): Single transform, or ``[gen_transform, seg_transform]``
            when ``is_base=True``.
        transform_torch (bool): If True, apply transform to torch tensor. Defaults to False.
        is_base (bool): If True, ``transform`` must be a length-2 list (generative + segmentation
            space). Defaults to False.
    """

    def __init__(self, data_dict, img_dir, transform=None, transform_torch:bool=False, is_base:bool=False):
        self.data_dict = data_dict
        self.img_dir = img_dir
        self.transform = transform
        self.transform_torch = True if is_base else transform_torch

        assert not(is_base) or (isinstance(self.transform, list) and len(self.transform) == 2), (
            '`transform` must be of type list and of length 2, where the first transformation ',
            'refers to the generative space and the second as required by the segmentation model'
        )

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """Return a (possibly dual-transformed) grayscale image tensor and image ID."""
        img = get_image_scaled(
            img_path=os.path.join(self.img_dir, f"{self.data_dict[idx]['img_file']}"), 
            transform=self.transform[0] if isinstance(self.transform, list) else self.transform, 
            transform_torch=self.transform_torch, convert_to='L', scale=1024
        )
        if isinstance(self.transform, list):
            img = torch.from_numpy(self.transform[1](img.numpy()))
        sample = {'img': img, 'img_id': re.sub(r'(\.png)|(\.jpg)', '', self.data_dict[idx]['img_file'].split('/')[-1])}
        return sample


def feature_scaling(img, center=1., factor=255./2., scale=1):
    """Scale image pixel values to approximately ``[-scale, scale]``.

    Adapted from PerceptualSimilarity ``util.util`` for classifier input normalization.

    Args:
        img (PIL.Image.Image or np.ndarray): Input image.
        center (float): Subtracted offset after division. Defaults to 1.0.
        factor (float): Divisor (typically ``255/2``). Defaults to ``255/2``.
        scale (float): Output range multiplier. Defaults to 1.

    Returns:
        np.ndarray: Float32 array in ``(C, H, W)`` layout.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 2: img = img[..., None]
    return (img.transpose(2, 0, 1).astype(np.float32) / factor - center) * scale


def get_image_scaled(img_path, transform, transform_torch:bool, convert_to:str='RGB', scale:int=1):
    """Load an image, apply :func:`feature_scaling`, and optionally a transform.

    Args:
        img_path (str): Path to the image file.
        transform (callable, optional): Additional transform to apply.
        transform_torch (bool): If True, pass a torch tensor to ``transform``.
        convert_to (str): PIL conversion mode. Defaults to ``'RGB'``.
        scale (int): Passed to :func:`feature_scaling`. Defaults to 1.

    Returns:
        torch.Tensor: Scaled (and optionally transformed) image tensor.
    """
    img = feature_scaling(Image.open(img_path).convert(convert_to), scale=scale)
    if transform:
        return transform(torch.from_numpy(img)) if transform_torch else torch.from_numpy(transform(img))
    else:
        return torch.from_numpy(img)

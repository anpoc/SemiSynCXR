"""
Image and mask utilities for CXR editing.

Provides helpers for decoding CheXmask RLE segmentations, resizing/cropping images and
bounding boxes, inferring mask dimensions, and applying spatial filters to binary masks.

Function groups:
    **Segmentation decoding**
        - :func:`rle2pixels` — decode CheXmask run-length-encoded masks to binary arrays.

    **Mask post-processing** (used by :mod:`maskGen`)
        - :func:`inferHW` — estimate half-width and half-height from a binary mask.
        - :func:`applyFilter` — Gaussian or generalized-Gaussian smoothing of mask(s).

    **Gaussian grid generation**
        - :func:`genGaussian1D`, :func:`createGridVector`, :func:`genGaussianGrid` — build
          separable 2D kernels for ``gengaussian`` filtering.

    **Image / bbox preprocessing**
        - :func:`crop`, :func:`pad`, :func:`cropPad` — compute scale factors and adjusted bbox
          coordinates for a target canvas size.
        - :func:`adjustBBox`, :func:`checkerBBox` — apply scale/offset and flag out-of-bounds boxes.
        - :func:`centerCrop` — center-crop or resize-then-crop images to a target size.

Consumers:
    - :mod:`maskGen` — :func:`rle2pixels`, :func:`inferHW`, :func:`applyFilter`.
    - :mod:`inpaintGen` — :func:`rle2pixels`.
"""

import torch
import numpy as np
import PIL.Image

from typing import Optional, Union
from scipy.ndimage import gaussian_filter

from torchvision.transforms.functional import center_crop, resize


def adjustBBox(metadata, dims):
    """Scale and center bounding boxes after a resize/pad operation.

    Expects ``metadata`` to contain columns ``x``, ``y``, ``w``, ``h``, ``adj_factor``,
    ``image_width``, and ``image_height``. Writes ``x_new``, ``y_new``, ``w_new``, ``h_new``.

    Args:
        metadata (pd.DataFrame): Annotation rows with original bbox and scale factor.
        dims (tuple[int, int]): Target canvas size as ``(width, height)``.
    """
    width, height = dims
    metadata['x_new'] = (metadata.x * metadata.adj_factor + 
        (width - metadata.image_width * metadata.adj_factor) / 2).astype(int)
    metadata['y_new'] = (metadata.y * metadata.adj_factor + 
        (height - metadata.image_height * metadata.adj_factor) / 2).astype(int)
    metadata['w_new'] = (metadata.w * metadata.adj_factor).astype(int)
    metadata['h_new'] = (metadata.h * metadata.adj_factor).astype(int)


def checkerBBox(metadata, dims, printflag=True):
    """Mark bounding boxes that fall outside the target canvas.

    Adds a boolean column ``use_bbox`` to ``metadata``.

    Args:
        metadata (pd.DataFrame): Must contain ``x_new``, ``y_new``, ``w_new``, ``h_new``.
        dims (tuple[int, int]): Target canvas size as ``(width, height)``.
        printflag (bool): If True, print the count of unusable boxes. Defaults to True.
    """
    width, height = dims
    metadata['use_bbox'] = (metadata.x_new >= 0) & (metadata.y_new >= 0) & \
        ((metadata.x_new + metadata.w_new) <= width) & ((metadata.y_new + metadata.h_new) <= height)
    if printflag: print(f'There are {(~metadata.use_bbox).sum()} non-usable bounding boxes.')


def crop(metadata, dims=(512,512), printflag=True):
    """Compute crop-style scale factors and adjusted bboxes for a target size.

    Uses ``max(width/image_width, height/image_height)`` so the image fills the canvas
    (center-crop behavior). Calls :func:`adjustBBox` and :func:`checkerBBox`.

    Args:
        metadata (pd.DataFrame): Annotation metadata with ``image_width``, ``image_height``,
            and original bbox columns.
        dims (tuple[int, int]): Target size ``(width, height)``. Defaults to ``(512, 512)``.
        printflag (bool): Passed to :func:`checkerBBox`. Defaults to True.
    """
    width, height = dims
    # Target dimension on smallest dimension
    metadata['adj_factor'] = metadata[['image_width', 'image_height']].apply(
        lambda x: max(width / x[0], height / x[1]), axis=1)
    # Adjusting bboxes and checking usability
    adjustBBox(metadata, dims)
    checkerBBox(metadata, dims, printflag)


def pad(metadata, dims=(512,512), printflag=True):
    """Compute pad-style scale factors and adjusted bboxes for a target size.

    Uses ``min(width/image_width, height/image_height)`` so the full image fits inside
    the canvas (letterbox behavior). Calls :func:`adjustBBox` and :func:`checkerBBox`.

    Args:
        metadata (pd.DataFrame): Annotation metadata with ``image_width``, ``image_height``,
            and original bbox columns.
        dims (tuple[int, int]): Target size ``(width, height)``. Defaults to ``(512, 512)``.
        printflag (bool): Passed to :func:`checkerBBox`. Defaults to True.
    """
    width, height = dims
    # Target dimension on smallest dimension
    metadata['adj_factor'] = metadata[['image_width', 'image_height']].apply(
        lambda x: min(width / x[0], height / x[1]), axis=1)
    # Adjusting bboxes and checking usability
    adjustBBox(metadata, dims)
    checkerBBox(metadata, dims, printflag)


def cropPad(metadata, dims=(512,512), printflag=True):
    """Hybrid crop-then-pad bbox adjustment for samples that fail a pure crop.

    Applies :func:`crop` first, then recomputes ``adj_factor`` with pad scaling for rows
    whose bounding boxes would fall outside the canvas.

    Args:
        metadata (pd.DataFrame): Annotation metadata.
        dims (tuple[int, int]): Target size ``(width, height)``. Defaults to ``(512, 512)``.
        printflag (bool): Passed to :func:`checkerBBox`. Defaults to True.
    """
    width, height = dims
    crop(metadata, dims=(512,512), printflag=False)
    use_bbox_filter = metadata.dicom_id.isin(metadata.loc[~metadata.use_bbox, 'dicom_id'].unique())
    metadata.loc[use_bbox_filter, 'adj_factor'] = metadata.loc[use_bbox_filter, 
        ['image_width', 'image_height']].apply(lambda x: min(width / x[0], height / x[1]), axis=1)
    adjustBBox(metadata, dims)
    checkerBBox(metadata, dims, printflag)


def centerCrop(img, target_dim=(512,512), printflag=False):
    """Center-crop an image to ``target_dim``.

    Supports ``numpy.ndarray`` (converted via torch), ``torch.Tensor``, and
    ``PIL.Image.Image``. For PIL images, resizes with the crop scale factor then crops
    the center region.

    Args:
        img (Union[np.ndarray, torch.Tensor, PIL.Image.Image]): Input image.
        target_dim (tuple[int, int]): Output size ``(width, height)``. Defaults to ``(512, 512)``.
        printflag (bool): Unused; kept for API compatibility. Defaults to False.

    Returns:
        Union[torch.Tensor, PIL.Image.Image]: Center-cropped image (type matches input path).
    """
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)
    if isinstance(img, torch.Tensor):
        img = center_crop(resize(img, size=target_dim[0]), output_size=target_dim)
    elif isinstance(img, PIL.Image.Image):
        ow, oh = img.size
        nw, nh = target_dim
        adj_factor = max(nw / ow, nh / oh)
        img = img.resize((int(ow * adj_factor), int(oh * adj_factor)))
        left = int((ow * adj_factor - nw)/2)
        top = int((oh * adj_factor - nh)/2)
        right = int((ow * adj_factor + nw)/2)
        bottom = int((oh * adj_factor + nh)/2)
        img = img.crop((left, top, right, bottom))
    return img


def applyFilter(img:list, filter_cfg:Optional[str], std:Union[list]=[1.], normalize:bool=True):
    """Apply a spatial filter to one or more binary masks and merge by pixel-wise maximum.

    Filter configuration string format: ``'<type>+<scale>'`` where ``type`` is
    ``'gaussian'`` or ``'gengaussian'``.

    Args:
        img (list): List of binary mask arrays (same shape).
        filter_cfg (Optional[str]): Filter spec, e.g. ``'gaussian+1'`` or ``'gengaussian+2'``.
            If None, returns the single mask or the element-wise maximum of multiple masks.
        std (Union[list, np.ndarray]): Per-mask standard deviation(s) for the filter kernel,
            typically bbox half-dimensions from :func:`inferHW`. Defaults to ``[1.]``.
        normalize (bool): If True, each filtered mask is divided by its peak before merging.
            Defaults to True.

    Returns:
        np.ndarray: Filtered mask array with the same shape as the inputs.

    Raises:
        ValueError: If ``filter_cfg`` specifies an unsupported filter type.
    """
    if filter_cfg is None:
        return img[0] if len(img) == 1 else np.maximum(*img)
    filter_type, arg_scale = filter_cfg.split('+')
    img_f = np.zeros_like(img[0])
    if filter_type == 'gaussian':
        for im, s in zip(img, std):
            img_aux = gaussian_filter(im, sigma=np.array(s) // float(arg_scale), mode='constant', truncate=3)
            img_f = np.maximum(img_aux / img_aux.max() if normalize else img_aux, img_f)
    elif filter_type == 'gengaussian':
        for im, s in zip(img, std):
            img_aux = genGaussianGrid(
                grid_center= [x[0] + s[j] for j, x in enumerate(np.where(im))], grid_shape=im.shape,
                sigma=np.array(s), beta=int(arg_scale), truncate=3
            )
            img_f = np.maximum(img_aux / img_aux.max() if normalize else img_aux, img_f)
    else:
        raise ValueError("Not implemented filter. Please use 'gaussian' or 'gengaussian'")
    return img_f


def createGridVector(base_vector, center, border, radius):
    """Embed a 1D kernel vector into a longer vector centered at ``center``.

    Pads with zeros when the kernel extends past image borders.

    Args:
        base_vector (np.ndarray): 1D filter kernel.
        center (int): Center index in the output vector.
        border (int): Length of the output vector (image dimension).
        radius (int): Half-width of the kernel support.

    Returns:
        np.ndarray: Padded 1D vector of length ``border``.
    """
    lzeros = center - radius
    rzeros = (border - 1) - (center + radius)
    vector = np.concatenate([
        np.zeros(max(lzeros, 0)),
        base_vector[max(-lzeros, 0):(2 * radius + 1) + min(rzeros, 0)],
        np.zeros(max(rzeros, 0))
    ])
    return vector


def genGaussian1D(std, beta, radius:int = 1):
    """Build a 1D generalized Gaussian kernel.

    Args:
        std (float): Scale parameter (related to standard deviation).
        beta (float): Shape exponent (``2`` for Gaussian, lower values for heavier tails).
        radius (int): Half-width of the kernel support in pixels. Defaults to 1.

    Returns:
        np.ndarray: Normalized 1D kernel of length ``2 * radius + 1``.
    """
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(- ((np.abs(x) / (np.sqrt(2) * std)) ** beta))
    phi_x = phi_x / phi_x.sum()
    return phi_x


def genGaussianGrid(grid_center, grid_shape, sigma, beta=2, truncate=3):
    """Build a separable 2D generalized Gaussian grid.

    Args:
        grid_center (list[int]): ``(row, col)`` center of the kernel in pixel coordinates.
        grid_shape (tuple[int, int]): Output grid shape ``(height, width)``.
        sigma (array-like): Standard deviations ``(sigma_y, sigma_x)`` or ``(sigma_row, sigma_col)``.
        beta (float): Generalized Gaussian exponent. Defaults to 2.
        truncate (float): Kernel radius as ``truncate * sigma`` per axis. Defaults to 3.

    Returns:
        np.ndarray: 2D kernel array of shape ``grid_shape``.
    """
    x = createGridVector(
        base_vector=genGaussian1D(sigma[0], beta, radius=int(sigma[0] * truncate)),
        center=grid_center[0], border=grid_shape[0], radius=int(sigma[0] * truncate)
    )
    y = createGridVector(
        base_vector=genGaussian1D(sigma[1], beta, radius=int(sigma[1] * truncate)),
        center=grid_center[1], border=grid_shape[1], radius=int(sigma[1] * truncate)
    )
    return x.reshape(-1,1) @ y.reshape(1,-1)


def inferHW(img):
    """Infer half-height and half-width from the support of a binary mask.

    Uses the first and last foreground pixel along each axis from :func:`numpy.where`.

    Args:
        img (np.ndarray): Binary mask array.

    Returns:
        np.ndarray: ``[half_h, half_w]`` as integers.
    """
    h, w = np.where(img)
    return np.array([(h[-1] - h[0]) // 2, (w[-1] - w[0]) // 2], dtype=int)


def rle2pixels(code, dim):
    """Decode a CheXmask run-length-encoded segmentation string to a binary mask.

    Args:
        code (str): Space-separated RLE string (alternating start index and run length,
            1-based start indices as in CheXmask exports).
        dim (tuple[int, int]): Image dimensions ``(height, width)``.

    Returns:
        np.ndarray: Binary mask of shape ``dim`` with dtype ``uint8``.
    """
    code = code.split()
    mask = np.zeros(np.prod(dim), dtype=np.uint8)
    for start, length in zip(code[0:-1:2], code[1::2]):
        mask[(int(start) - 1):(int(start) - 1 + int(length))] = 1
    return mask.reshape(dim)

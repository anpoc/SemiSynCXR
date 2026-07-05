"""
Spatial mask generation for CXR editing.

Generates anatomically plausible binary masks for inpainting by sampling bounding box positions
and dimensions from fitted prior distributions (see :mod:`priorDistrGen`). Masks are constrained
to anatomical regions determined by prompt-based restrictions (see :mod:`promptGen`) and
lung/heart segmentations from CheXmask.

Pipeline:
    1. Parse the prompt to determine laterality and anatomical region restrictions.
    2. Load lung/heart segmentations for the target image.
    3. Sample bounding box center and dimensions from the fitted prior distributions,
       normalized relative to the enclosing anatomical structure.
    4. Convert normalized coordinates to pixel space and generate a binary mask.
    5. Optionally apply a spatial filter (e.g., Gaussian smoothing) and save.

Usage:
    python maskGen.py --config ./configs/pipeline.json --img <dicom_id> --label <label>
        [--prompt <prompt>]

Configuration:
    Requires ``pipeline.json`` and a per-label mask config at ``mask_segment.config_label``:
        - ``metadata``: Paths to general metadata and CheXmask segmentation CSV shards.
        - ``priors.savepath``: Directory with ``distr_center.pickle`` and ``distr_wh.pickle``.
        - ``mask_segment.savepath``: Output directory, formatted with ``version``.
        - Per-label settings: ``location``, ``wh.minimum``, ``wh.bandwidth``.

Outputs:
    Written under ``mask_segment.savepath``:
        - ``{mask_id}.png`` — single-lung or Cardiomegaly mask.
        - ``{mask_id}_{idx}.png`` — bilateral masks (one file per lung).
        - ``info.csv`` — append-only metadata (UUID, label, prompt, sampling probabilities).

Coordinate conventions:
    Centers and sizes are sampled in percentage coordinates (0–100) relative to the enclosing
    lung bounding box. :func:`promptGen.get_restrictions` returns allowed center regions as
    ``((x_lo, y_lo), (x_hi, y_hi))``. Left-lung x values are mirrored before pixel conversion
    so priors fitted on right-lung data apply to both sides.
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import cv2
import uuid

from ast import literal_eval
from csv import DictWriter
from typing import Any, Optional, TypeAlias, Union

from utils.distributionUtils import sample_jointdistr, sample_distr
from utils.imageUtils import applyFilter, rle2pixels, inferHW
from priorDistrGen import import_distributions
from promptGen import get_restrictions, BBox


PixelBBox: TypeAlias = list[int]
"""Bounding box as [x_min, y_min, x_max, y_max] in pixel coordinates."""

SegmentationMap: TypeAlias = list[PixelBBox]
"""List of bounding boxes for anatomical structures."""

ImageDim: TypeAlias = list[int]
"""Image dimensions as [height, width]."""

SamplingProb: TypeAlias = tuple[Optional[float], float]
"""Sampling probability tuple (prob_xy, prob_wh) where prob_xy may be None for Cardiomegaly."""


def _safe_eval_bbox(value: Any, field_name: str = "bbox") -> PixelBBox:
    """Safely parse a bounding box value from metadata.

    Args:
        value: Either a string-encoded list or an already-parsed list.
        field_name: Name of the field for error messages.

    Returns:
        PixelBBox: Parsed bounding box as [x_min, y_min, x_max, y_max].

    Raises:
        ValueError: If the value cannot be parsed as a valid bounding box.
    """
    if isinstance(value, list):
        return value
    try:
        result = literal_eval(value)
        if not isinstance(result, list) or len(result) != 4:
            raise ValueError(f"Expected list of 4 coordinates, got {type(result).__name__}")
        return result
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse {field_name}: {value!r}") from e


class Mask():
    """Mask generator for spatially constrained CXR inpainting.

    Loads prior distributions and per-label configuration at initialization, then generates
    binary masks for a given image and pathology label by sampling bounding box parameters
    from the priors.

    Attributes:
        cfg_mask (dict): Per-pathology mask configuration (symmetry, location, minimum
            dimensions, bandwidth).
        metadata (dict): File paths for metadata CSVs (general, segmentation).
        distrs (dict): Fitted prior distributions for center and width/height.
        save_path (str): Directory for saving generated masks and metadata CSV.
        save_field_names (list[str]): Column names for the mask metadata CSV.
    """

    def __init__(self, config_path:str):
        """Initialize the mask generator from a pipeline configuration file.

        Args:
            config_path (str): Path to ``pipeline.json``.
        """
        with open(config_path, 'r') as f:
            cfg_pipe = json.load(f)
        with open(cfg_pipe['mask_segment']['config_label'], 'r') as f:
            self.cfg_mask = json.load(f)
        # General
        self.metadata = cfg_pipe['metadata']
        # Distributions
        self.distrs = import_distributions(cfg_pipe['priors']['savepath'])
        # Masks
        self.save_path = cfg_pipe['mask_segment']['savepath'].format(cfg_pipe['version'])
        os.makedirs(self.save_path, exist_ok=True)
        self.save_field_names = ['mask_id', 'mask_base_id', 'mask_prob', 'mask_filter', 'img_id', 'label', 'prompt']


    def get_bbox(
        self,
        label: str,
        xy_exclude: tuple[float, float, float] = (0, 0, 0),
        xy_bounds: BBox = ((0, 0), (100, 100)),
        wh_bounds: tuple[float, float, float, float] = (0, 0, 0, 0),
        unit_equiv: np.ndarray = np.array([1, 1]),
        ntries: int = 3
    ) -> tuple[Optional[np.ndarray], np.ndarray, SamplingProb]:
        """Sample a bounding box (center + dimensions) from the fitted prior distributions.

        Handles three pathology-specific sampling strategies:
            - **Cardiomegaly**: Samples only width (cardiothoracic ratio).
            - **Pleural Effusion**: Samples y-center and height; x-center fixed at 50%.
            - **All others**: Jointly samples (x, y) center and (w, h), rejecting samples
              that overlap the heart exclusion zone or violate bounds.

        Args:
            label (str): Pathology label (e.g. ``'Atelectasis'``, ``'Cardiomegaly'``).
            xy_exclude (tuple): Heart exclusion zone as
                ``(x_right_bound, y_top_bound, y_bottom_bound)`` in percentage coordinates.
            xy_bounds (tuple): Allowed center range as ``((x_min, y_min), (x_max, y_max))``.
            wh_bounds (tuple): Maximum overshoot beyond the segmentation boundary as
                ``(left, top, right, bottom)`` in percentage coordinates.
            unit_equiv (np.ndarray): Conversion factor ``100 / seg_diam`` per axis.
            ntries (int): Maximum rejection sampling attempts. Defaults to 3.

        Returns:
            tuple: ``(xy, wh, prob)`` where ``xy`` is the sampled center (or None for
            Cardiomegaly), ``wh`` is ``[w, h]``, and ``prob`` is ``(prob_xy, prob_wh)``.

        Raises:
            ValueError: If all ``ntries`` attempts are exhausted.
        """
        get_ub = lambda center, whb: 2 * np.minimum(np.subtract(center, whb[:2]), 100 - np.add(center, whb[2:]))
        if label == 'Cardiomegaly':
            w_lb = self.cfg_mask[label]['wh']['minimum'][0]
            w, prob_w = sample_distr(self.distrs['wh'][label]['x'], (w_lb, 100), unit_equiv[0])
            return None, np.array([w, 0]), (None, prob_w)
        elif label == 'Pleural Effusion':
            h_lb = self.cfg_mask[label]['wh']['minimum'][1]
            while ntries > 0:
                y, prob_y = sample_distr(
                    self.distrs['center'][label][0]['y'], (xy_bounds[0][1], xy_bounds[1][1]), unit_equiv[1]
                )
                h_ub = get_ub([0, y], wh_bounds)[1]
                ntries = -1 if h_ub > h_lb else ntries - 1
            if ntries == 0:
                raise ValueError('All number of tries used')
            h, prob_h = sample_distr(self.distrs['wh'][label]['y'], (max(h_lb, 2 * (100 - y)), h_ub), unit_equiv[1])
            return np.array([50., y]), np.array([100., h]), (prob_y, prob_h)
        else:
            wh_lb = self.cfg_mask[label]['wh']['minimum']
            ntries *= 2
            while ntries > 0:
                xy, prob_xy = sample_jointdistr(self.distrs['center'][label][0], xy_bounds, unit_equiv)
                wh_ub = get_ub(xy, wh_bounds)
                if (np.array(wh_ub) < np.array(wh_lb)).any():
                    ntries -= 1
                    continue
                wh, prob_wh = sample_jointdistr(self.distrs['wh'][label], (wh_lb, wh_ub), unit_equiv)
                xy_ends = [xy[0] + wh[0] // 2, xy[1] - wh[1] // 2, xy[1] + wh[1] // 2]
                ntries = -1 if (np.subtract(xy_ends, xy_exclude) * np.array([1, -1, 1]) > 0).any() else ntries - 1
            if ntries == 0:
                raise ValueError('All number of tries used')
            return xy, wh, (prob_xy, prob_wh)


    def collect_segs(self, vals: list[str], dim: ImageDim) -> SegmentationMap:
        """Decode RLE-encoded segmentations into bounding boxes.

        Args:
            vals (list): RLE-encoded segmentation strings for each anatomical structure.
            dim (list): Image dimensions as ``[height, width]``.

        Returns:
            list: Bounding boxes as ``[x_min, y_min, x_max, y_max]`` in pixels.

        Raises:
            ValueError: If any segmentation string cannot be decoded.
        """
        seg_map = []
        for val in vals:
            try:
                seg = np.where(rle2pixels(val, dim=dim))
            except (ValueError, TypeError) as e:
                raise ValueError(f'Not available segmentations on CheXmask: {e}')
            seg_map.append([seg[1].min(), seg[0].min(), seg[1].max(), seg[0].max()])
        return seg_map


    def get_segmentation(
        self,
        img_id: str,
        locs: list[str],
        locs_extra: list[str] = [],
        vals: Optional[dict] = None
    ) -> tuple[SegmentationMap, SegmentationMap, ImageDim]:
        """Load anatomical segmentation bounding boxes for a given image.

        Retrieves lung and auxiliary (heart or full-lung) segmentations either from a
        pre-loaded ``vals`` row or by searching partitioned segmentation CSVs on disk.

        Args:
            img_id (str): DICOM ID of the target image.
            locs (list): Primary structures (e.g. ``['Right Lung']`` or both lungs).
            locs_extra (list): Auxiliary structures (``['Heart']`` or both lungs for
                Cardiomegaly). Defaults to ``[]``.
            vals (Optional[dict]): Pre-loaded metadata row. Defaults to None.

        Returns:
            tuple: ``(locs_map, extra_map, img_dim)`` — primary bboxes, auxiliary bboxes,
            and ``[height, width]``.
        """
        if vals is None:
            try:
                vals = pd.read_csv(self.metadata['general'], usecols=['dicom_id', 'Height', 'Width'] + locs + locs_extra)\
                    .set_index('dicom_id').loc[0]
            except (KeyError, FileNotFoundError):
                for f in os.listdir(self.metadata['segmentation']):
                    if not f.endswith('.csv'):
                        continue
                    start, end = f.split('.csv')[0].split('_')[1:3]
                    if start <= img_id <= end:
                        temp = pd.read_csv(f"{self.metadata['segmentation']}{f}", 
                            usecols=['dicom_id', 'Height', 'Width'] + locs + locs_extra)
                        vals = temp[temp.dicom_id == img_id].iloc[0]
                        break
                img_dim = vals[['Height', 'Width']].tolist() 
                locs_map = self.collect_segs(vals[locs].tolist(), img_dim)
                extra_map = self.collect_segs(vals[locs_extra].tolist(), img_dim)
                return locs_map, extra_map, img_dim

        img_dim = [int(vals['Height']), int(vals['Width'])]
        locs_map = [_safe_eval_bbox(vals[i], i) for i in locs]
        extra_map = [_safe_eval_bbox(vals[i], i) for i in locs_extra]
        return locs_map, extra_map, img_dim
        

    def import_base_mask(self, mask_base_id: str) -> list[np.ndarray]:
        """Load previously saved base mask(s) from disk.

        Args:
            mask_base_id (str): UUID of the base mask to load.

        Returns:
            list: Binary masks as float arrays with values in ``[0, 1]``.
        """
        i = 0
        mask_base = []
        while os.path.isfile(f'{self.save_path}{mask_base_id}_{i}.png'):
            mask_base.append(cv2.imread(f'{self.save_path}{mask_base_id}_{i}.png', cv2.IMREAD_GRAYSCALE) / 255)
            i += 1
        if len(mask_base) == 0:
            mask_base.append(cv2.imread(f'{self.save_path}{mask_base_id}.png', cv2.IMREAD_GRAYSCALE) / 255)
        return mask_base


    def postprocess(self, img_id:str, label:str, mask_base:Optional[list]=None, mask_base_id:Optional[str]=None, 
        mask_filter:Optional[str]=None, bbox_hw:Optional[list]=None, bbox_prob:Optional[list]=None, prompt:str=''):
        """Apply optional spatial filtering to base masks and save with metadata.

        Args:
            img_id (str): DICOM ID of the source image.
            label (str): Pathology label.
            mask_base (Optional[list]): Pre-computed binary base masks.
            mask_base_id (Optional[str]): UUID of a previously saved base mask.
            mask_filter (Optional[str]): Filter name (e.g. ``'gaussian'``). If None, no filtering.
            bbox_hw (Optional[list]): Bounding box ``[h, w]`` per mask for filter std.
            bbox_prob (Optional[list]): Sampling probabilities for metadata logging.
            prompt (str): Radiology prompt used for generation.

        Returns:
            str: UUID of the saved (possibly filtered) mask.
        """
        assert (mask_base is None) != (mask_base_id is None), 'Provide either mask or mask_id'
        if mask_base_id is not None:
            if mask_filter is None:
                return mask_base_id
            mask_base = self.import_base_mask(mask_base_id)
        assert all([set(np.unique(m)) == set([0, 1]) for m in mask_base]), 'Provide non filtered masks as mask_base'
        if bbox_hw is None:
            bbox_hw = [inferHW(m) for m in mask_base]
        mask = applyFilter(mask_base, mask_filter, std=bbox_hw, normalize=True)
        mask_id = str(uuid.uuid4())
        self.save_mask(mask, mask_id, mask_base_id, bbox_prob, mask_filter, img_id, label, prompt)
        return mask_id


    def save_mask(self, mask:Union[list, np.ndarray], mask_id:str, mask_base_id:Optional[str], mask_prob:Optional[list],
        mask_filter:Optional[str], img_id:str, label:str, prompt:str='', save_info:bool=True):
        """Save mask image(s) and optionally append metadata to ``info.csv``.

        Args:
            mask (Union[list, np.ndarray]): Mask(s) with values in ``[0, 1]``.
            mask_id (str): UUID for the mask file name(s).
            mask_base_id (Optional[str]): UUID of the unfiltered base mask.
            mask_prob (Optional[list]): Sampling probabilities for metadata tracking.
            mask_filter (Optional[str]): Name of the applied filter.
            img_id (str): DICOM ID of the source image.
            label (str): Pathology label.
            prompt (str): Radiology prompt used.
            save_info (bool): Whether to append a row to ``info.csv``. Defaults to True.
        """
        if save_info:
            row = {'mask_id': mask_id, 'mask_base_id': mask_base_id, 'mask_prob': mask_prob,
            'mask_filter': mask_filter, 'img_id': img_id, 'label': label, 'prompt': prompt}
            with open(f'{self.save_path}info.csv', 'a+') as f:
                dictwriter = DictWriter(f, fieldnames=self.save_field_names)
                if f.tell() == 0: dictwriter.writeheader()
                dictwriter.writerow(row)
        if isinstance(mask, list):
            for idx, m in enumerate(mask):
                cv2.imwrite(f'{self.save_path}{mask_id}_{idx}.png', m * 255)
        else:
            cv2.imwrite(f'{self.save_path}{mask_id}.png', mask * 255)


    def main(
        self,
        img_id: str,
        label: str,
        segmentations: Optional[dict] = None,
        prompt: str = ''
    ) -> str:
        """Generate a spatial mask for a given image and pathology label.

        End-to-end pipeline: parse prompt restrictions, load segmentations, sample bounding
        boxes from priors, convert to pixel coordinates, and save binary mask(s).

        For **Cardiomegaly**, width is derived from the cardiothoracic ratio (sampled width
        × thoracic diameter), centered on the lung midpoint. For **other labels**, center
        and dimensions are sampled relative to the lung segmentation, with heart exclusion
        for left-lung placements and x-mirroring for left-side sampling.

        Args:
            img_id (str): DICOM ID of the source image.
            label (str): Target pathology label.
            segmentations (Optional[dict]): Pre-loaded segmentation metadata row.
            prompt (str): Radiology prompt for anatomical restriction parsing.

        Returns:
            str: UUID of the generated mask.

        Raises:
            ValueError: If bounding box sampling fails after all retries.
            AssertionError: If Cardiomegaly lung order is incorrect.
        """
        is_cardiomegaly = label == 'Cardiomegaly'
        # Getting segmentations
        x_side, xy_bounds = get_restrictions(prompt, invert_left=True)
        x_locs = [np.random.choice(self.cfg_mask[label]['location'])] if x_side == 'any' \
            else (self.cfg_mask[label]['location'] if x_side == 'both' else [f'{x_side.capitalize()} Lung']) 
        seg_pos, seg_aux, img_dim = self.get_segmentation(
            img_id, x_locs, 
            locs_extra=self.cfg_mask['Lungs']['location'] if is_cardiomegaly else self.cfg_mask['Heart']['location'], 
            vals=segmentations
        )
        img_height, img_width = img_dim
        # Creating bounding boxes and mask
        masks = []
        bbox_probs = []
        bbox_hws = []
        for idx, seg in enumerate(seg_pos):
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            seg_diam = np.array([seg[2] - seg[0], seg[3] - seg[1]])
            seg_min = np.array([seg[0], seg[1]])
            if is_cardiomegaly:
                _, bbox_wh, bbox_prob = self.get_bbox(label, unit_equiv=100/seg_diam)
                bbox_xy = seg_min + seg_diam // 2
                w_adj = (seg_aux[1][2] - seg_aux[0][0]) * bbox_wh[0] / 100
                assert w_adj > 0, 'Please provide the lungs in the right order'
                bbox_wh = np.array([w_adj // 2, (seg_diam[1] * w_adj / seg_diam[0]) // 2], dtype=int)
            else:
                seg_side = int(x_locs[idx] == 'Left Lung')
                wh_bounds = np.maximum(
                    -np.array(seg[0:2] + [img_width - seg[2], img_height - seg[3]]) / np.tile(seg_diam, 2) * 100, 
                    self.cfg_mask[label]['wh']['bandwidth']
                )
                if seg_side:
                    xy_exclude = (
                        (1 - (seg_aux[0][2] - seg[0]) / seg_diam[0]) * 100, 
                        (seg_aux[0][1] - seg[1]) / seg_diam[1] * 100, 
                        (seg_aux[0][3] - seg[1]) / seg_diam[1] * 100
                    )
                    wh_bounds[2], wh_bounds[0] = wh_bounds[0], wh_bounds[2]
                else:
                    xy_exclude = (0, 0, 0)
                bbox_xy, bbox_wh, bbox_prob =self.get_bbox(
                    label, xy_exclude=xy_exclude, xy_bounds=xy_bounds[-seg_side], wh_bounds=wh_bounds, 
                    unit_equiv=100/seg_diam
                )
                bbox_xy[0] = abs(seg_side * 100 - bbox_xy[0])
                bbox_xy = (seg_min + bbox_xy * seg_diam / 100).astype(int)
                bbox_wh = (bbox_wh * seg_diam / 100 / 2).astype(int)
            mask[
                (bbox_xy[1] - bbox_wh[1]):(bbox_xy[1] + bbox_wh[1] + 1), 
                (bbox_xy[0] - bbox_wh[0]):(bbox_xy[0] + bbox_wh[0] + 1)
            ] = 1
            masks.append(mask)
            bbox_hws.append(bbox_wh[::-1])
            bbox_probs.append(bbox_prob)
        # Postprocessing
        mask_id = self.postprocess(
            img_id, label, masks, mask_filter=None, prompt=prompt, bbox_hw=bbox_hws, bbox_prob=bbox_probs
        )
        self.save_mask(masks, mask_id, None, None, None, '', '', save_info=False)
        return mask_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script that creates a mask for a given image and label.')
    parser.add_argument('--config', required=False, type=str, default='./configs/pipeline.json', 
        help='Path to the pipeline configuration file (pipeline.json).')
    parser.add_argument('--img', required=True, type=str, help='Image ID.')
    parser.add_argument('--label', required=True, type=str, help='Label or target label.')
    parser.add_argument('--prompt', required=False, type=str, default='', help='Prompt to be used.')
    args = parser.parse_args()

    mask = Mask(args.config)
    mask.main(img_id=args.img, label=args.label, prompt=args.prompt)

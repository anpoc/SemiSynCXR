"""
Inpainting generation pipeline for CXR editing.

Orchestrates end-to-end generation of synthetically edited chest X-rays by combining mask
generation (see :mod:`maskGen`), prompt sampling (see :mod:`promptGen`), and diffusion-based
inpainting. Supports multiple inpainting backends (RoentGen, RadEdit, DiffEdit, Blended Latent
Diffusion, X-Real) with configurable inference parameters.

Pipeline:
    1. Validate CLI inputs and prepare output directories.
    2. Build a row-indexed job dictionary (images, masks, labels, prompts) via
       :func:`create_data_dict`.
    3. Generate or filter masks when needed (:func:`get_mask`).
    4. Run the selected diffusion inpainting pipeline in batches (:func:`main`).
    5. Save inpainted images and append metadata to ``info.csv``.

Usage:
    python inpaintGen.py --config ./configs/pipeline.json [--nsamples N] [--img ID]
        [--label LABEL] [--prompt TEXT] [--mask MASK_ID]

Configuration:
    Requires ``pipeline.json`` with ``inpaint``, ``mask_segment``, ``metadata``, and ``imgsize``
    sections. Key ``inpaint`` fields: ``pipeline``, ``modelpath``, ``labels``, ``label_as_prompt``,
    ``num_inference_steps``, ``guidance_scale``, ``strength``, ``batch_size``, ``savepath``.

Outputs:
    Written under ``inpaint.savepath`` formatted with ``version``:
        - ``{inpaint_id}.png`` — inpainted image per job.
        - ``concat/{inpaint_id}.png`` — optional side-by-side grid (source, mask, output).
        - ``info.csv`` — append-only metadata (UUID, pipeline, paths, label, prompt).

Note:
    Source image selection has been pre-processed. The filtering logic (PA view, erect positioning,
    no support devices, no bounding box overlap) is documented below for reference::

        info = pd.read_csv(metadata['general'])
        bbox_ids = pd.read_csv(metadata['bbox'], usecols=['dicom_id']).dicom_id.tolist()
        normal_ids = pd.read_csv(metadata['labels'])
        normal_ids = normal_ids.loc[
            (normal_ids['No Finding'] == 1) & (normal_ids['Support Devices'] != 1),
            ['subject_id', 'study_id']
        ]
        classifier_ids = pd.read_csv(metadata['classification'])['img_path'].tolist()
        info = info[
            (~info['dicom_id'].isin(bbox_ids)) &
            (info['ViewPosition'] == 'PA') &
            (info['PatientOrientationCodeSequence_CodeMeaning'] == 'Erect') &
            info['img_path'].isin(classifier_ids)
        ].merge(normal_ids, on=['subject_id', 'study_id'], how='inner')

Dependencies:
    - :mod:`maskGen` — :class:`maskGen.Mask` for mask generation and post-processing.
    - :mod:`promptGen` — :func:`promptGen.generate_prompt` for sampled report phrases.
    - :mod:`utils.datadictUtils` — :class:`CXRDataset` for batched image/mask loading.
    - :mod:`pipelines` — custom diffusers pipeline wrappers.
"""

import argparse, os, uuid, re
import json, pickle, csv
import pandas as pd
import numpy as np
import torch

from math import floor
from typing import Optional, Union
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from torchvision.transforms.functional import to_pil_image

import pipelines
from maskGen import Mask
from promptGen import generate_prompt
from utils.imageUtils import rle2pixels
from utils.datadictUtils import CXRDataset

from multiprocessing import Pool

from PIL import Image


global mask_obj


def check_inputs(img_id, label, prompt, mask_id, n_gen, save_path):
    """Validate CLI argument combinations and create output directories.

    Args:
        img_id: Image path or list of paths, or None to sample from metadata.
        label: Target pathology label(s), or None to use config defaults.
        prompt: Optional explicit prompt(s); must align with ``label`` count.
        mask_id: Pre-existing mask UUID(s), or None to generate new masks.
        n_gen (int): Number of samples per label when sampling images.
        save_path (str): Root directory for inpainting outputs.

    Returns:
        int: ``0`` when all of ``img_id``, ``label``, and ``mask_id`` are None
        (batch mode from existing inpaint metadata CSV).
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + 'concat/', exist_ok=True)
    if img_id is None and label is None and mask_id is None:
        if prompt is not None: print('The prompt(s) will not be taken into account as no label(s) are given.')
        return 0

    if label is None: n_label = 0
    elif type(label) == str: n_label = 1
    else: n_label = len(label)

    if type(img_id) == list:
        assert (len(img_id) == n_gen) or (len(img_id) == n_gen * n_label), 'The number of ' + \
            'provided images should equal the number of generations per prompt or in total.'
    if type(mask_id) == list:
        assert len(mask_id) == n_gen * n_label, 'The number of provided masks should equal the ' + \
            'number of generated images.'
    else:
        assert type(img_id) == str and n_label == 1, 'The number of provided images and prompts ' + \
            'should be also one as only one mask is provided.'

    if prompt is not None:
        assert n_label == 1 if type(prompt) == str else n_label == len(prompt), \
            'The number of given labels and prompts should be the same.'


def get_extremes(code, dim):
    """Decode an RLE segmentation and return its axis-aligned bounding box extremes.

    Args:
        code (str): CheXmask RLE-encoded segmentation string.
        dim (tuple): Image dimensions ``(height, width)``.

    Returns:
        list or float: ``[x_min, y_min, x_max, y_max]``, or ``np.nan`` if decoding fails.
    """
    try:
        y, x = np.where(rle2pixels(code, dim))
        return [x.min(), y.min(), x.max(), y.max()]
    except:
        return np.nan


def get_segmentations(img_id:list, process_segmentation_path:str, raw_segmentation_path:str):
    """Load lung/heart bounding boxes for a batch of DICOM IDs.

    Reads pre-parsed bboxes from the general metadata CSV when available; falls back to
    partitioned raw segmentation CSV shards and RLE decoding via :func:`get_extremes`.

    Args:
        img_id (list): DICOM IDs to look up.
        process_segmentation_path (str): Path to general metadata CSV with parsed bboxes.
        raw_segmentation_path (str): Directory of CheXmask segmentation CSV shards.

    Returns:
        dict: ``{dicom_id: {Left Lung, Right Lung, Heart, Height, Width}}``.
    """
    cols = ['Left Lung', 'Right Lung', 'Heart']
    df_segs = pd.read_csv(process_segmentation_path).set_index('dicom_id')
    seg_all = df_segs.loc[img_id][cols].map(eval).merge(
        df_segs.loc[img_id][['Height', 'Width']], left_index=True, right_index=True
    ).to_dict(orient='index')
    img_id = list(set(img_id).difference(seg_all.keys()))
    if img_id:
        img_id_sorted = sorted(img_id)
        files = [x for x in os.listdir(raw_segmentation_path) if x.endswith('.csv')]
        files_sorted = sorted(files)
        id_base = 0
        for f in files_sorted:
            _, end = f.split('.csv')[0].split('_')[1:3]
            if img_id_sorted[id_base] <= end:
                seg = pd.read_csv(f'{raw_segmentation_path}{f}', usecols=['dicom_id', 'Height', 'Width'] + cols)
                seg = seg[seg['dicom_id'].isin(img_id)]
                for col in cols:
                    seg[col] = seg[[col, 'Height', 'Width']].apply(
                        lambda x: get_extremes(x.iloc[0], dim=x.iloc[1:].values), axis=1)
                seg_all.update(seg.set_index('dicom_id').to_dict(orient='index'))
                id_base += seg.shape[0]
            if len(seg_all) == len(img_id):
                break
    return seg_all


def mask_prostprocess_wrapper(args):
    """Multiprocessing worker for :meth:`maskGen.Mask.postprocess`.

    Args:
        args (tuple): ``(img_id, label, mask_base_id, mask_filter, prompt)``.

    Returns:
        str: UUID of the filtered mask.
    """
    img_id, label, mask_base_id, mask_filter, prompt = args
    return mask_obj.postprocess(img_id, label, mask_base_id=mask_base_id, mask_filter=mask_filter, prompt=prompt)


def get_mask(mask_base_id, img_id, label, prompt, metadata):
    """Generate base masks and apply configured spatial filtering in parallel.

    When ``mask_base_id`` is None, calls :meth:`maskGen.Mask.main` per image. Otherwise
    reuses provided base mask UUIDs. Filtering uses ``cfg_pipe['mask_segment']['filter']``
    via a :class:`multiprocessing.Pool`.

    Args:
        mask_base_id: None, a single mask UUID string, or a list of base mask UUIDs.
        img_id (list): DICOM IDs, one per generation job.
        label (list): Pathology labels.
        prompt (list): Radiology prompts.
        metadata (dict): Metadata paths with keys ``'general'`` and ``'segmentation'``.

    Returns:
        list[str]: Filtered mask UUIDs, one per job.
    """
    if mask_base_id is None:
        mask_base_id = []
        segs = get_segmentations(img_id, metadata['general'], metadata['segmentation'])
        for i, l, p in zip(img_id, label, prompt):
            mask_base_id.append(mask_obj.main(i, l, segs[i], p))
    elif type(mask_base_id) == str:
        mask_base_id = [mask_base_id]

    with Pool(os.cpu_count()) as pool:
        mask_id = pool.map(mask_prostprocess_wrapper, zip(img_id, label, mask_base_id, 
            [cfg_pipe['mask_segment']['filter']] * len(img_id), prompt))
    return mask_id


def get_img(img_name:Optional[Union[str,list]], n_gen:int, metadata:dict, rstate:int=1):
    """Resolve source image paths for inpainting jobs.

    Args:
        img_name (Optional[Union[str, list]]): Explicit image path(s), or None to sample
            from metadata.
        n_gen (int): Number of images to sample when ``img_name`` is None.
        metadata (dict): Metadata file paths with key ``'general'``.
        rstate (int): Random state for reproducible sampling. Defaults to 1.

    Returns:
        list[str]: Image file paths (e.g. ``['p10/p10000032/.../img.jpg']``).
    """
    if img_name is None:
        img_name = pd.read_csv(metadata['general']).dropna(subset=['Heart', 'Right Lung', 'Left Lung'])\
            .sample(n=n_gen, replace=False, random_state=rstate)['img_path'].to_list()
    elif type(img_name) is str:
        img_name = [img_name]
    return img_name


def get_prompt(label: list, label_as_prompt:bool=True):
    """Generate radiology prompts for a list of pathology labels.

    Args:
        label (list): Pathology labels.
        label_as_prompt (bool): If True, use the label string directly as the prompt.
            If False, sample from :func:`promptGen.generate_prompt`. Defaults to True.

    Returns:
        list[str]: One prompt string per label.
    """
    if label_as_prompt:
        return label
    prompt = []
    for l in label:
        prompt.append(generate_prompt(l))
    return prompt


def get_mask_kwargs(mask:torch.Tensor, pipeline_name:str=''):
    """Build pipeline-specific keyword arguments for the mask input.

    RadEdit expects ``edit_mask`` and ``keep_mask`` (complementary); all other pipelines
    expect ``mask_image``.

    Args:
        mask (torch.Tensor): Binary mask tensor with values in ``[0, 1]``.
        pipeline_name (str): Name of the inpainting pipeline class. Defaults to ``''``.

    Returns:
        dict: Keyword arguments to pass to the pipeline's ``__call__`` method.
    """
    mask_kwargs = {}
    if 'RadEdit' in pipeline_name:
        mask_kwargs['edit_mask'] = mask
        mask_kwargs['keep_mask'] = 1 - mask
    else:
        mask_kwargs['mask_image'] = mask
    return mask_kwargs


def create_data_dict(img, mask, label, prompt, n_samples, cfg_pipe):
    """Build a row-indexed dictionary of inpainting job specifications.

    If ``img``, ``mask``, and ``label`` are all None and a pre-existing inpainting metadata
    CSV exists at ``cfg_pipe['metadata']['inpaint']``, loads that CSV directly. Otherwise
    assembles image paths, mask IDs, labels, and prompts into a new :class:`pandas.DataFrame`.

    Args:
        img: Source image path(s) or None to sample from metadata.
        mask: Pre-existing mask UUID(s) or None to generate new masks.
        label: Target pathology label(s) or None to read from config.
        prompt: Explicit prompt(s); currently unused when building from scratch (prompts
            come from :func:`get_prompt`).
        n_samples (int): Number of samples per label from CLI ``--nsamples``.
        cfg_pipe (dict): Full pipeline configuration.

    Returns:
        dict: Row-indexed job dict consumed by :class:`utils.datadictUtils.CXRDataset`.
    """
    data_path = cfg_pipe['metadata'].get('inpaint', None)
    if img is None and mask is None and label is None and os.path.isfile(data_path):
        data = pd.read_csv(data_path)
        data['mask_id'] = get_mask(data['mask_id'].to_list(), data['img_id'].to_list(), data['label'].to_list(), 
            data['prompt'].to_list(), cfg_pipe['metadata'])
    else:
        if label is None:
            n_samples = cfg_pipe['inpaint']['nsamples']
            label = cfg_pipe['inpaint']['labels'] * n_samples
        elif type(label) == str:
            label = [label] * n_samples
        img_name = get_img(img, len(label), cfg_pipe['metadata'])
        img_id = list(map(lambda x: re.sub(r'(\.png)|(\.jpg)', '', x.split('/')[-1]), img_name))
        prompt = get_prompt(label, cfg_pipe['inpaint']['label_as_prompt'])
        mask_id = get_mask(mask, img_id, label, prompt, cfg_pipe['metadata'])
        data = pd.DataFrame(dict(zip(['img_id', 'mask_id', 'label', 'prompt', 'img_file'], 
            [img_id, mask_id, label, prompt, img_name])))
    return data.to_dict('index')


def save_batch_inpaint(inpainted, img, mask, label, prompt, ids, pipeline_name, save_path, save_concat):
    """Save a batch of inpainted images and append metadata rows to ``info.csv``.

    Args:
        inpainted (list): PIL images returned by the diffusion pipeline.
        img (torch.Tensor, optional): Source image batch (for concat grids).
        mask (torch.Tensor, optional): Mask batch (for concat grids).
        label (list): Pathology labels per image.
        prompt (list): Prompts per image.
        ids (tuple): ``(img_file_paths, mask_ids)`` identifier lists.
        pipeline_name (str): Name of the inpainting pipeline used.
        save_path (str): Output directory for PNGs and ``info.csv``.
        save_concat (bool): If True and ``img``/``mask`` are provided, save a 3-panel grid
            under ``save_path/concat/``.
    """
    n_gen = len(inpainted)
    inpainted_id = [str(uuid.uuid4()) for _ in range(n_gen)]
    field_names = ['inpaint_id', 'pipeline', 'img_file', 'mask_id', 'label', 'prompt']
    rows = list(map(list, zip(inpainted_id, [pipeline_name] * n_gen, ids[0], ids[1], label, prompt)))
    with open(f"{save_path}info.csv", 'a+') as f:
        writer = csv.writer(f)
        if f.tell() == 0: writer.writerow(field_names)
        writer.writerows(rows)

    for idx in range(n_gen):
        inpainted[idx].save(f'{save_path}{inpainted_id[idx]}.png')
        if save_concat and img is not None and mask is not None:
            make_image_grid([to_pil_image(img[idx]), to_pil_image(mask[idx]), inpainted[idx]], 
                rows=1, cols=3).save(f'{save_path}concat/{inpainted_id[idx]}.png')


def latents_to_rgb(latents):
    """Approximate RGB visualization of diffusion latents (debug helper).

    Applies a fixed linear projection from latent channels to RGB. Used by
    :func:`decode_tensors` for step-wise debugging.

    Args:
        latents (torch.Tensor): Latent tensor from a diffusion callback.

    Returns:
        PIL.Image.Image: RGB preview image.
    """
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )
    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)

def decode_tensors(pipe, step, timestep, callback_kwargs):
    """Diffusion callback that saves a latent RGB preview at each denoising step.

    Args:
        pipe: Diffusion pipeline instance (unused).
        step (int): Current denoising step index.
        timestep: Current scheduler timestep (unused).
        callback_kwargs (dict): Must contain ``'latents'`` key.

    Returns:
        dict: Unmodified ``callback_kwargs``.
    """
    latents = callback_kwargs["latents"]
    image = latents_to_rgb(latents[0])
    image.save(f"./{step}.png")

    return callback_kwargs


def main(
    data_dict:dict, pipeline_name, img_size:int, img_path:str='./image/', mask_path:str='./mask/', 
    model_path:str='./models/roentgen', device:str='cuda', num_inference_steps:int=50, 
    guidance_scale:float=7.5, strength:float=1.0, use_negative_prompt:bool=False, stop_mask_pct:float=0.0,
    batch_size:int=32, disable_safety_checker:bool=True, 
    save_kwargs:dict={'save_path': './inpainting/', 'save_concat': False}
) -> None:
    """Run batched diffusion inpainting over a job dictionary.

    Instantiates the configured pipeline from :mod:`pipelines`, builds a
    :class:`CXRDataset` / :class:`DataLoader`, and iterates batches. Pipeline-specific
    scheduler and keyword arguments are set for DiffEdit, Blended, X-Real, and RadEdit.

    Args:
        data_dict (dict): Row-indexed jobs from :func:`create_data_dict`.
        pipeline_name (str): Pipeline class name (e.g. ``'RoentGenPipeline'``).
        img_size (int): Target square image size after resize and center crop.
        img_path (str): Root directory for source CXR images.
        mask_path (str): Root directory for saved mask PNGs.
        model_path (str): HuggingFace model or local checkpoint path.
        device (str): Torch device string. Defaults to ``'cuda'``.
        num_inference_steps (int): Denoising steps. Defaults to 50.
        guidance_scale (float): Classifier-free guidance scale. Defaults to 7.5.
        strength (float): Inpaint strength / blending ratio (pipeline-specific). Defaults to 1.0.
        use_negative_prompt (bool): If True, add a negative finding prompt. Defaults to False.
        stop_mask_pct (float): Fraction of steps after which mask constraint is released.
            Defaults to 0.0.
        batch_size (int): DataLoader batch size. Defaults to 32.
        disable_safety_checker (bool): Disable diffusers safety checker. Defaults to True.
        save_kwargs (dict): Passed to :func:`save_batch_inpaint` (``save_path``, ``save_concat``).

    Returns:
        None
    """
    # Create dataset and dataloader
    dataset = CXRDataset(data_dict, img_dir=img_path, mask_dir=mask_path, 
        transform=Compose([ToTensor(), Resize(img_size), CenterCrop(img_size)]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Instanciate pipeline
    if pipeline_name != 'RadEditPipeline':
        inpaint = getattr(pipelines, pipeline_name).from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        inpaint.enable_model_cpu_offload()
        if disable_safety_checker:
            inpaint.safety_checker = None
    else:
        inpaint = DiffusionPipeline.from_pipe(getattr(pipelines, pipeline_name), custom_pipeline="microsoft/radedit")\
            .to(device)

    if pipeline_name == 'RadEditPipeline':
        extra_kwargs = {
            "eta": 1.0,
            "weights": [3.5, guidance_scale] if use_negative_prompt else [guidance_scale],
            "invert_prompt": generate_prompt('No Finding') if use_negative_prompt else '',
            "skip_ratio": 1.0 - strength,
            "num_inference_steps": int(200 * (num_inference_steps / 75)),
        }
    else:
        extra_kwargs = {
            "strength": strength,
            "eta": 1.0,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "return_dict": False,
        }
        if pipeline_name == 'DiffEditPipeline':
            inpaint.scheduler = pipelines.schedulers.DDIMModScheduler.from_config(inpaint.scheduler.config)
            extra_kwargs.update({
                "guidance_scale_inv": 1.0,
                "prompt_inv": '',
            })
        elif pipeline_name == 'BlendedPipeline':
            inpaint.scheduler = pipelines.schedulers.DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
            # 'strength' corresponds to 'blending_percentage' in the original code
            extra_kwargs.update({
                "strength": strength,
                "eta": 0.0
            })
        elif pipeline_name == 'XRealPipeline':
            inpaint.scheduler = pipelines.schedulers.DDPMScheduler(
                beta_start=0.0015,
                beta_end=0.0295,
                beta_schedule="scaled_linear",
            )
    extra_kwargs['stop_mask_step'] = floor(extra_kwargs['num_inference_steps']  * stop_mask_pct)

    for _, sample_batched in enumerate(dataloader):
        if use_negative_prompt and (pipeline_name != 'RadEditPipeline'):
            extra_kwargs.update({
                "negative_prompt": [generate_prompt('No Finding')] * len(sample_batched['prompt'])
            })
        output, _ = inpaint(
            prompt=sample_batched['prompt'],
            image=sample_batched['img'],
            #callback_on_step_end=decode_tensors,
            **get_mask_kwargs(sample_batched['mask'], pipeline_name),
            **extra_kwargs
        )
        save_batch_inpaint(output, pipeline_name=pipeline_name, **sample_batched, **save_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script that inpaints an image.')
    parser.add_argument('--config', required=False, type=str, default='./configs/pipeline.json', 
        help='Path to the pipeline configuration file (pipeline.json).')
    parser.add_argument('--nsamples', required=False, type=int, default=1, help='No. of samples per label or prompt.')
    parser.add_argument('--img', required=False, type=Optional[str], default=None, help='Image name.')
    parser.add_argument('--label', required=False, type=Optional[str], default=None, help='Target label.')
    parser.add_argument('--prompt', required=False, type=Optional[str], default=None, help='Prompt.')
    parser.add_argument('--mask', required=False, type=Optional[str], default=None, help='Mask ID.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_pipe = json.load(f)
    mask_obj = Mask(args.config)

    check_inputs(
        img_id=args.img, label=args.label, prompt=args.prompt, mask_id=args.mask, n_gen=args.nsamples,
        save_path=cfg_pipe['inpaint']['savepath'].format(cfg_pipe['version'])
    )
    data_dict = create_data_dict(args.img, args.mask, args.label, args.prompt, args.nsamples, cfg_pipe)

    main(
        data_dict=data_dict,
        pipeline_name=cfg_pipe['inpaint']['pipeline'],
        img_size=cfg_pipe['imgsize'],
        img_path=cfg_pipe['inpaint']['imgpath'],
        mask_path=cfg_pipe['mask_segment']['savepath'].format(cfg_pipe['version']),
        model_path=cfg_pipe['inpaint']['modelpath'],
        device=cfg_pipe['inpaint']['device'],
        num_inference_steps=cfg_pipe['inpaint']['num_inference_steps'],
        guidance_scale=cfg_pipe['inpaint']['guidance_scale'],
        strength=cfg_pipe['inpaint']['strength'],
        stop_mask_pct=cfg_pipe['inpaint'].get('stop_mask_pct', 0),
        use_negative_prompt=cfg_pipe['inpaint']['use_negative_prompt'],
        batch_size=cfg_pipe['inpaint']['batch_size'],
        save_kwargs={
            'save_path': cfg_pipe['inpaint']['savepath'].format(cfg_pipe['version']), 'save_concat': False
        }
    )
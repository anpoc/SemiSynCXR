# SemiSynCXR: Semi-Synthetic Localization Datasets for Radiological Findings on Chest X-Rays

A framework for generating semi-synthetic chest X-ray (CXR) localization datasets with realistic pathological findings. SemiSynCXR addresses the scarcity of labeled bounding box data for radiological findings by combining anatomically-informed spatial priors with text-conditioned diffusion inpainting to generate localized radilogical finding augmentations. The generated data can be used to train object detection models for automatic radilogical finding localization.

**Authors:** Andrea Posada, Johannes Brandt, Friederike Jungmann, Maria Posada, Daniel Rueckert, Martin J. Menten, Felix meissen, Philip Müller

## Abstract

While large datasets for chest X-ray (CXR) finding classification are widely available, datasets for finding localization are scarce. Curating these localization datasets is costly and time-intensive, requiring manual annotation by medical experts, which often results in them being small and limited in scope. To overcome this, we introduce *SemiSynCXR*, a framework designed to automatically generate semi-synthetic localization datasets. *SemiSynCXR* operates by inpainting specific radiological findings into real, healthy CXRs at anatomically plausible locations, which allows for the output of both the edited image and the ground-truth bounding box for each finding. *SemiSynCXR*-generated CXRs effectively augment existing localization datasets, yielding relative $mAP_{10:70}$ gains of up to 11\% on in-domain and 21\% on out-of-domain data, thereby mitigating data scarcity and improving generalization. Comprehensive quantitative and qualitative evaluations show that our framework achieves an overall AUROC of 0.78 and $mAP_{10:70}$ of 0.45, comparable to fully synthetic benchmarks. These results confirm that the generated findings are realistic and accurately localized, establishing *SemiSynCXR* as a practical solution for the generation of CXR finding localization datasets. Code available at the [SemiSynCXR GitHub Repository](https://github.com/anpoc/SemiSynCXR).

## Overview

1. **Prior Distribution Learning** (`priorDistrGen.py`) — Learns spatial priors for pathology locations from annotated bounding boxes (MS-CXR dataset), normalized relative to anatomical structures (lungs, heart).

2. **Prompt Generation** (`promptGen.py`) — Provides probability-weighted radiology prompts derived from MIMIC-CXR reports, with automatic anatomical region restriction parsing.

3. **Mask Generation** (`maskGen.py`) — Samples anatomically plausible binary masks by drawing bounding box parameters from fitted prior distributions, constrained by lung/heart segmentations.

4. **Editing** (`inpaintGen.py`) — Orchestrates diffusion-based editing using multiple banckends.

5. **Evaluation** (`inpaintEval.py`) — Evaluates generated images via classification, objcet detection, CLIP-score, and FID scoring.

## Supported Pathologies

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Lung Opacity
- Pleural Effusion
- Pneumothorax

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Access to MIMIC-CXR dataset (for training/evaluation)

### Setup

```bash
# Clone the repository
git clone https://github.com/perezvon/SemiSynCXR.git
cd SemiSynCXR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Pre-trained Models

Download the required models and place them in the `src/models/` directory:

- **RoentGen**: Stable Diffusion fine-tuned on CXR images
- **CXR-CLIP**: For CLIP-score evaluation
- **Segmentation models**: For anatomical structure extraction

## Project Structure

```
SemiSynCXR/
├── src/
│   ├── configs/               # Pipeline configuration files
│   │   └── pipeline.json      # Main configuration template
│   ├── pipelines/             # Diffusion inpainting pipelines
│   │   ├── simpleAfter.py     # Blended Latent Diffusion (post-blend)
│   │   ├── simpleBefore.py    # X-Real style (pre-blend)
│   │   ├── radEdit.py         # RadEdit pipeline
│   │   ├── diffEdit.py        # DiffEdit pipeline
│   │   └── schedulers/        # Custom schedulers
│   ├── utils/                 # Utility modules
│   │   ├── imageUtils.py      # Image processing utilities
│   │   ├── distributionUtils.py  # Distribution fitting/sampling
│   │   ├── datadictUtils.py   # PyTorch Dataset classes
│   │   ├── evaluationUtils.py # Evaluation metrics
│   │   └── cxrclip/           # CXR-CLIP integration
│   ├── priorDistrGen.py       # Prior distribution generation
│   ├── promptGen.py           # Prompt generation and parsing
│   ├── maskGen.py             # Spatial mask generation
│   ├── inpaintGen.py          # Inpainting generation pipeline
│   └── inpaintEval.py         # Evaluation module
├── results/                   # Output directory (generated)
├── requirements.txt           # Python dependencies
└── README.md
```

## Usage

### 1. Configure the Pipeline

Create or modify a configuration file in `src/configs/`:

```json
{
    "version": "v1",
    "imgsize": 512,
    "metadata": {
        "general": "/path/to/mimic_metadata.csv",
        "segmentation": "/path/to/segmentation/",
        "bbox": "/path/to/mscxr_bbox.csv"
    },
    "priors": {
        "savepath": "./results/priors/",
        "from_raw": true
    },
    "mask_segment": {
        "config_label": "./configs/masks.json",
        "savepath": "./results/masks{}/"
    },
    "inpaint": {
        "pipeline": "InpaintAfterPipeline",
        "modelpath": "./models/roentgen",
        "labels": ["Atelectasis", "Cardiomegaly"],
        "nsamples": 100
    }
}
```

### 2. Generate Prior Distributions

```bash
cd src
python priorDistrGen.py --config ./configs/pipeline.json
```

### 3. Generate Masks

```bash
python maskGen.py --config ./configs/pipeline.json --img <dicom_id> --label "Atelectasis"
```

### 4. Run Inpainting

```bash
# Generate synthetic pathology images
python inpaintGen.py --config ./configs/pipeline.json --nsamples 100

# With specific parameters
python inpaintGen.py --config ./configs/pipeline.json \
    --img <dicom_id> \
    --label "Pleural Effusion" \
    --prompt "Right pleural effusion."
```

### 5. Evaluate Results

```bash
# Classification evaluation
python inpaintEval.py --config ./configs/pipeline.json --mode classification_gen

# CLIP-score evaluation
python inpaintEval.py --config ./configs/pipeline.json --mode clipscore
```

## Pipeline Components

### Inpainting Pipelines

| Pipeline | Description | Best For |
|----------|-------------|----------|
| `InpaintAfterPipeline` | Blended Latent Diffusion with post-denoising blend | Smooth transitions |
| `InpaintBeforePipeline` | X-Real style with pre-denoising blend | Anatomical consistency |
| `RadEditPipeline` | RadEdit with edit/keep mask separation | Medical image editing |
| `DiffEditPipeline` | DiffEdit with guided inversion | Controlled editing |

### Mask Filters

- `gaussian+N` — Standard Gaussian blur with sigma=bbox_dim/N
- `gengaussian+N` — Generalized Gaussian with adjustable beta parameter

## Configuration Options

### Inpainting Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_inference_steps` | Denoising steps | 50 |
| `guidance_scale` | Classifier-free guidance scale | 7.5 |
| `strength` | Denoising strength (0-1) | 1.0 |
| `use_negative_prompt` | Use "No Finding" as negative prompt | false |
| `stop_mask_pct` | Fraction of steps to stop mask application | 0.0 |

## Data Requirements

### MIMIC-CXR Metadata

The pipeline expects preprocessed MIMIC-CXR metadata with:
- Patient/study/DICOM identifiers
- View position (PA/AP)
- Patient orientation
- Image paths

### Segmentation Data

Anatomical segmentations (CheXmask format) with RLE-encoded masks for:
- Left Lung
- Right Lung
- Heart

### MS-CXR Annotations

Bounding box annotations for pathology prior extraction.

## Evaluation Metrics

- **Classification Accuracy**: TorchXRayVision DenseNet/ResNet classifiers
- **CLIP Score**: CXR-CLIP image-text similarity
- **FID Score**: Fréchet Inception Distance
- **LPIPS/SSIM**: Perceptual and structural similarity

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{popp2026semisynCXR,
    title={Semi-Synthetic Localization Datasets for Radiological Findings on Chest X-Rays},
    author={Popp, Alexander and Perez-Leiva, Antonio and Hering, Alessa and M{\"u}ller, Philip and Braren, Rickmer and Kaissis, Georgios and Rueckert, Daniel},
    booktitle={Medical Imaging with Deep Learning (MIDL)},
    year={2026}
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0).

## Acknowledgments

### Datasets
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) — Source images and radiology reports
- [MS-CXR](https://physionet.org/content/ms-cxr/0.1/) — Bounding box annotations for spatial prior learning
- [CheXmask](https://physionet.org/content/chexmask-cxr-segmentation-data/) — Anatomical segmentations (lung/heart)
- [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/) — Scene graph annotations for prompt generation
- [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) — External evaluation dataset

### Models and Libraries
- [RoentGen](https://arxiv.org/abs/2211.12737) — CXR-specialized Stable Diffusion model
- [RadEdit](https://arxiv.org/abs/2406.10777) — Medical image editing pipeline
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) — CXR classification models
- [CXR-CLIP](https://arxiv.org/abs/2310.13292) — Medical image-text similarity
- [Diffusers](https://github.com/huggingface/diffusers) — Diffusion model library

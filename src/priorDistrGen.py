"""
Prior Distribution Generation for CXR Editing.

Generates prior spatial distributions for radiological findings based on bounding box
annotations from the MS-CXR dataset. These priors model where pathologies typically appear
relative to anatomical structures (lungs, heart), and are used downstream by the mask
generation module to produce anatomically plausible editing masks.

Pipeline:
    1. Load and preprocess bounding box metadata (fix outlier annotations).
    2. Extract raw prior statistics (center positions and width/height ratios) normalized
       relative to lung or thoracic dimensions.
    3. Fit 1D or 2D probability distributions to the extracted priors.
    4. Serialize the fitted distributions for use by ``maskGen.py``.

Usage:
    python priorDistrGen.py [--config ./configs/pipeline.json] [--force]

Configuration:
    Requires two JSON config files loaded at startup into module-level globals
    ``cfg_pipe`` and ``cfg_mask``:
        - ``pipeline.json``: File paths (metadata, save directories) and distribution
          fitting parameters (1D/2D distribution families).
        - Label-level mask config (referenced by ``pipeline.json``): Per-pathology
          settings including symmetry, distribution type, and center constraints.

Outputs:
    Written under ``cfg_pipe['priors']['savepath']`` (default ``../results/priors/``):
        - ``prior_center.pickle``, ``prior_wh.pickle``: Raw normalized statistics per label.
        - ``distr_center.pickle``, ``distr_wh.pickle``: Fitted parametric distributions.
        - ``fits1D/``: Per-label fit summaries (CSV) from :mod:`utils.distributionUtils`.
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import json
import os

from utils import distributionUtils


# Pipeline and per-label mask configs.
cfg_pipe = None
cfg_mask = None


def preprocess_meta(metadata):
    """Preprocess bounding box metadata by correcting outlier annotations.

    Applies manual corrections to specific DICOM samples where anatomical bounding boxes
    (Left Lung, Right Lung) are incorrectly annotated — e.g., lung boundaries extending
    beyond the image or heart region. Also removes samples that are too corrupted to fix.

    Corrections include:
        - Clamping left lung lower bounds to the heart boundary.
        - Clamping right lung lower bounds relative to the left lung.
        - Adjusting crop height (``h``) and vertical offset (``y``) for samples where the
          lung region is disproportionate to the crop.
        - Removing irrecoverable samples.

    Outlier samples are identified by ``dicom_id/x`` (crop x-coordinate) to disambiguate
    multiple crops per study.

    Args:
        metadata (pd.DataFrame): DataFrame with columns ``dicom_id``, ``x``, ``y``, ``w``,
            ``h``, ``Left Lung``, ``Right Lung``, ``Heart`` (the latter three as
            string-encoded ``[x1, y1, x2, y2]`` coordinate lists).

    Returns:
        pd.DataFrame: Cleaned metadata with corrected bounding boxes and problematic samples
            removed.
    """
    metadata['temp'] = metadata['dicom_id'] + '/' + metadata['x'].astype(str)
    # Left lung: clamp lower bound to heart or crop boundary
    for i in ['65ef31a2-e080f853-c5c75be5-2246e4e8-105fffb1/1829', 'be39a244-3124c478-f202e009-29c3ea27-ce5a3256/1497']:
        row = metadata[metadata['temp'] == i]
        ll = eval(row['Left Lung'].item())
        h = eval(row['Heart'].item())
        ll[3] = min(row['y'].item() + row['h'].item(), h[3])
        metadata.loc[metadata['dicom_id'] == row['dicom_id'].item(), 'Left Lung'] = str(ll)

    # Right lung: clamp lower bound relative to left lung
    for i in [
        '81ec392d-d7d7b085-57108530-40a1ba37-91cce411/119', '1ad2ef88-52d312dc-98ea0159-8f4a5670-b4912f9f/173', 
        '7f920625-025c2064-91d6ef6d-85da9c3f-d9dc4e04/566', 'adb67a2b-a1473b01-b1004ae8-e612d0c9-1c2b5e3b/109', 
        'd0b50a99-ae808cea-c678ca94-dbbb2401-09f7194e/104'
    ]:
        row = metadata[metadata['temp'] == i]
        ll = eval(row['Left Lung'].item())
        rl = eval(row['Right Lung'].item())
        rl[3] = min(row['y'].item() + row['h'].item(), ll[3] - max(0, ll[1] - rl[1]))
        metadata.loc[metadata['dicom_id'] == row['dicom_id'].item(), 'Right Lung'] = str(rl)

    # Both lungs: correct left and right lung bboxes for paired crops
    for i in [('6b7eea34-85cbf22f-b5f190dd-e66aae9a-bdcebc99/114', '6b7eea34-85cbf22f-b5f190dd-e66aae9a-bdcebc99/1694')]:
        ## Left
        row = metadata[metadata['temp'] == i[1]]
        ll = eval(row['Left Lung'].item())
        h = eval(row['Heart'].item())
        ll[3] = min(row['y'].item() + row['h'].item(), h[3] + 1000)
        metadata.loc[metadata['dicom_id'] == row['dicom_id'].item(), 'Left Lung'] = str(ll)
        ## Right
        row = metadata[metadata['temp'] == i[0]]
        ll = eval(row['Left Lung'].item())
        rl = eval(row['Right Lung'].item())
        rl[3] = min(row['y'].item() + row['h'].item(), ll[3] - max(0, ll[1] - rl[1]))
        metadata.loc[metadata['dicom_id'] == row['dicom_id'].item(), 'Right Lung'] = str(rl)

    # Reduce crop height to match lung extent
    for i in ['10a99391-935979a4-5bb5dd0d-0b46a553-e0cffd4b/1979']:
        row = metadata[metadata['temp'] == i]
        ll = eval(row['Left Lung'].item())
        rl = eval(row['Right Lung'].item())
        metadata.loc[metadata['temp'] == i, 'h'] = 2 * (ll[3] - row['y'].item())

    # Increase crop height to match lung extent
    for i in ['dc56ff3e-b5074c00-5dbb2327-b6a2250d-8db5a157/353', 'ed730ed6-e391f6a6-55e52913-a66b2844-da028e10/382']:
        row = metadata[metadata['temp'] == i]
        ll = eval(row['Left Lung'].item())
        rl = eval(row['Right Lung'].item())
        ymax = row['y'].item() + row['h'].item()
        metadata.loc[metadata['temp'] == i, 'y'] = ymax - 2 * (ymax - rl[3])
        metadata.loc[metadata['temp'] == i, 'h'] = 2 * (ymax - rl[3])

    # Delete irrecoverable samples
    metadata = metadata[~metadata['temp'].isin([
        '1e0e30d4-4b8c4ec6-8600f00a-085a260b-8d16c141/173', 'a8c08cbf-15ac0dac-b76a40a0-dab826c7-18015767/436',
        'a8c08cbf-15ac0dac-b76a40a0-dab826c7-18015767/156',
        '146df8f2-587b432d-3151e370-fcba82ce-23b7e6ff/654', 'ae91431f-db70a388-3f5f2a99-3de56e9b-ae0f2119/1957',
        '412226a0-fe97f725-576b21f8-3d8d4fd3-ea7cfa4a/315', '267cc457-a4573bd8-2f401d19-1257fb65-04115df7/300',
        '5335699d-de67abb4-a2143b68-30ca5a5d-f5410238/168'
    ])].drop(columns=['temp'])

    return metadata


def extract_priors(data, saveflag=True, savepath: str = '../results/priors/'):
    """Extract raw prior statistics from bounding box annotations.

    For each pathology label, computes:
        - **Center priors**: The (x, y) center of each bounding box as a percentage of the
          enclosing lung structure dimensions. For symmetric pathologies, left-lung centers
          are mirrored to align with right-lung coordinates.
        - **Width/height priors**: Bounding box dimensions as a percentage of the enclosing
          lung structure. For Cardiomegaly, width is the cardiothoracic ratio
          (cardiac diameter / thoracic diameter).

    Cardiomegaly contributes only to width/height priors (no center prior), since location
    is defined by the heart bounding box rather than lung-relative coordinates.

    Requires module-level ``cfg_mask`` for per-label symmetry settings.

    Args:
        data (pd.DataFrame): Preprocessed metadata from :func:`preprocess_meta`.
        saveflag (bool): Whether to serialize priors to disk. Defaults to True.
        savepath (str): Directory to save ``prior_center.pickle`` and ``prior_wh.pickle``.
            Defaults to ``'../results/priors/'``.

    Returns:
        dict: Top-level keys ``'center'`` and ``'wh'``, each mapping pathology labels to
            dicts of ``{'x': [...], 'y': [...]}`` or ``{'w': [...], 'h': [...]}``.
    """
    priors = dict(zip(['center', 'wh'], [{}, {}]))
    cols = ['x', 'y', 'w', 'h', 'image_width', 'image_height', 'Right Lung', 'Left Lung', 'Heart']
    groups = {k: g.to_dict('records') for k, g in data.groupby('category_name')[cols]}
    for label, bboxes in groups.items():
        is_cardiomegaly = label == 'Cardiomegaly'
        is_symmetric = cfg_mask[label].get('center', {}).get('symmetry', True)
        prior_center = {
            'x': [],
            'y': []
        }
        prior_wh =  {
            'w': [],
            'h': []
        }
        for bb in bboxes:
            if not is_cardiomegaly:
                prior_center = center_prior(prior_center, bb, is_symmetric=is_symmetric)
            prior_wh = wh_prior(prior_wh, bb, is_cardiomegaly=is_cardiomegaly)
        if not is_cardiomegaly:
            priors['center'][label] = prior_center
        priors['wh'][label] = prior_wh
    if saveflag:
        os.makedirs(savepath, exist_ok=True)
        for k in priors.keys():
            with open(f'{savepath}/prior_{k}.pickle', 'wb') as f:
                pickle.dump(priors[k], f, protocol=pickle.HIGHEST_PROTOCOL)
    return priors


def center_prior(prior, bb, is_symmetric=True):
    """Compute the normalized center position of a bounding box relative to its lung.

    Determines which lung (left or right) the bounding box center is closest to, then
    expresses the center as a percentage of that lung's width and height. For symmetric
    pathologies, left-lung x-coordinates are mirrored (``100 - x``); for asymmetric ones,
    left-lung x-coordinates are offset (``200 + x``) to preserve side information.

    Args:
        prior (dict): Accumulator with keys ``'x'`` and ``'y'``, each a list of percentage
            values.
        bb (dict): Single bounding box record with keys ``'x'``, ``'y'``, ``'w'``, ``'h'``,
            ``'Right Lung'``, and ``'Left Lung'`` (string-encoded ``[x1, y1, x2, y2]``
            coordinates).
        is_symmetric (bool): If True, mirror left-lung coordinates. Defaults to True.

    Returns:
        dict: Updated ``prior`` with appended (x, y) percentages.
    """
    x = bb['x'] + bb['w'] // 2
    y = bb['y'] + bb['h'] // 2
    rlung = eval(bb['Right Lung'])
    llung = eval(bb['Left Lung'])
    idxlung = np.argmin([abs(x - rlung[0]), abs(x - rlung[2]), abs(x - llung[0]), abs(x - llung[2])])
    structure = rlung if idxlung <= 1 else llung
    x_pct = (x - structure[0]) / (structure[2] - structure[0]) * 100
    y_pct = (y - structure[1]) / (structure[3] - structure[1]) * 100
    if idxlung > 1:
        x_pct = 100 - x_pct if is_symmetric else 200 + x_pct
    prior['x'].append(x_pct)
    prior['y'].append(y_pct)
    return prior


def wh_prior(prior, bb, is_cardiomegaly=False):
    """Compute the normalized width and height of a bounding box relative to its lung.

    For non-Cardiomegaly findings, width and height are expressed as percentages of the
    enclosing lung structure dimensions. For Cardiomegaly, width is the cardiothoracic
    ratio (cardiac diameter / thoracic diameter × 100) and height is the aspect ratio
    (h / w × 100).

    Args:
        prior (dict): Accumulator with keys ``'w'`` and ``'h'``, each a list of percentage
            values.
        bb (dict): Single bounding box record with keys ``'x'``, ``'w'``, ``'h'``,
            ``'Right Lung'``, ``'Left Lung'``, and ``'Heart'`` (string-encoded
            ``[x1, y1, x2, y2]`` coordinates).
        is_cardiomegaly (bool): If True, use cardiothoracic ratio. Defaults to False.

    Returns:
        dict: Updated ``prior`` with appended (w, h) percentages.
    """
    rlung = eval(bb['Right Lung'])
    llung = eval(bb['Left Lung'])
    if is_cardiomegaly:
        heart = eval(bb['Heart'])
        cardiac_diam = heart[2] - heart[0]
        thoracic_diam = max(rlung[2] - llung[0], llung[2] - rlung[0])
        prior['w'].append(cardiac_diam / thoracic_diam * 100)
        prior['h'].append(bb['h'] / bb['w'] * 100)
    else:
        x = bb['x'] + bb['w'] // 2
        idxlung = np.argmin([abs(x - rlung[0]), abs(x - rlung[2]), abs(x - llung[0]), abs(x - llung[2])])
        structure = rlung if idxlung <= 1 else llung
        prior['w'].append(bb['w'] / (structure[2] - structure[0]) * 100)
        prior['h'].append(bb['h'] / (structure[3] - structure[1]) * 100)
    return prior


def import_priors(readpath: str = '../results/priors/'):
    """Load previously extracted raw priors from disk.

    Args:
        readpath (str): Directory containing ``prior_center.pickle`` and ``prior_wh.pickle``.
            Defaults to ``'../results/priors/'``.

    Returns:
        dict: Keys ``'center'`` and ``'wh'``, each mapping pathology labels to
            ``{'x': [...], 'y': [...]}`` or ``{'w': [...], 'h': [...]}``.
    """
    priors = {}
    for prior in ['center', 'wh']:
        with open(f'{readpath}/prior_{prior}.pickle', 'rb') as f:
            priors[prior] = pickle.load(f)
    return priors


def create_distributions(priors, saveflag=True, savepath: str = '../results/priors/'):
    """Fit parametric distributions to the extracted priors.

    For each pathology:
        - **Center distributions**: Fits 1D (independent x, y) or 2D (joint) distributions.
          For symmetric pathologies, a single distribution is fit to mirrored data; for
          asymmetric ones, separate distributions are fit per side (right-lung data vs.
          left-lung data offset by 300). Side assignment uses ``x < 150`` as the right-lung
          filter.
        - **Width/height distributions**: Fits 1D or 2D distributions to the (w, h) priors.

    Distribution families are specified in ``cfg_pipe['priors']['1D_distr']`` and
    ``cfg_pipe['priors']['2D_distr']``. Per-label 1D vs 2D and symmetry settings come from
    ``cfg_mask``. Fitted distributions are serialized incrementally — the pickle file is
    overwritten after each label is processed.

    Args:
        priors (dict): Output from :func:`extract_priors` with top-level keys ``'center'``
            and ``'wh'``.
        saveflag (bool): Whether to serialize fitted distributions. Defaults to True.
        savepath (str): Directory to save pickle files and ``fits1D/`` fit statistics.
            Defaults to ``'../results/priors/'``.

    Returns:
        None
    """
    path_ = savepath + '/fits1D/'
    os.makedirs(path_, exist_ok=True)
    distr_1D = cfg_pipe['priors']['1D_distr']
    distr_2D = cfg_pipe['priors']['2D_distr']
    distrs = {}
    k = 'center'
    for label, prior in priors[k].items():
        distrs[label] = []
        is_symmetric = cfg_mask[label][k].get('symmetry', True)
        is_distr_1D = [cfg_mask[label][k].get('distr') == '1D'] if is_symmetric \
            else [i == '1D' for i in cfg_mask[label][k].get('distr')]
        x = np.array(prior['x'])
        y = np.array(prior['y'])
        side_filter = x < 150
        for idx_side in range(abs(int(is_symmetric) - 2)):
            distrs[label] += [distributionUtils.fit_multivar(
                data={
                    'x': list(x[side_filter] if idx_side == 0 else 300 - x[side_filter]),
                    'y': list(y[side_filter])
                },
                independent=is_distr_1D[idx_side],
                distributions=distr_1D if is_distr_1D[idx_side] else distr_2D,
                savestats=True,
                savepath=f'{path_}{k}_{label}_{idx_side}'
            )]
            side_filter = ~side_filter
        if saveflag:
            with open(f'{savepath}/distr_{k}.pickle', 'wb') as f:
                pickle.dump(distrs, f, protocol=pickle.HIGHEST_PROTOCOL)
    k = 'wh'
    for label, prior in priors[k].items():
        is_distr_1D = cfg_mask[label][k].get('distr') == '1D'
        distrs[label] = distributionUtils.fit_multivar(
            data=prior,
            independent=is_distr_1D,
            distributions=distr_1D if is_distr_1D else distr_2D,
            savestats=True,
            savepath=f'{path_}{k}_{label}'
        )
        if saveflag:
            with open(f'{savepath}/distr_{k}.pickle', 'wb') as f:
                pickle.dump(distrs, f, protocol=pickle.HIGHEST_PROTOCOL)


def import_distributions(readpath: str = '../results/priors/'):
    """Load previously fitted distributions from disk.

    Args:
        readpath (str): Directory containing ``distr_center.pickle`` and ``distr_wh.pickle``.
            Defaults to ``'../results/priors/'``.

    Returns:
        dict: Keys ``'center'`` and ``'wh'``, each mapping pathology labels to fitted
            distribution objects from :mod:`utils.distributionUtils`.
    """
    distrs = {}
    for distr in ['center', 'wh']:
        with open(f'{readpath}/distr_{distr}.pickle', 'rb') as f:
            distrs[distr] = pickle.load(f)
    return distrs


def main(force=False):
    """Main entry point for prior distribution generation.

    Loads bounding box metadata from the path specified in ``cfg_pipe``, joins with view
    position metadata, and applies the following filters:
        - Exclude Cardiomegaly annotations from AP (anteroposterior) views, as the
          cardiothoracic ratio is unreliable in AP projections.
        - Include only erect patient positioning.
        - Exclude Pneumonia annotations (handled separately from Consolidation).

    After preprocessing, attempts to load existing fitted distributions. If none are found,
    either extracts raw priors from scratch (if ``cfg_pipe['priors']['from_raw']`` is True)
    or loads previously saved priors, then fits distributions.

    Args:
        force (bool): If True, regenerate priors even when valid distribution pickles exist.

    Requires module-level ``cfg_pipe`` and ``cfg_mask`` to be set before invocation.
    """
    # Setting up
    metadata = pd.read_csv(cfg_pipe['metadata']['bbox'])
    viewpos = pd.read_csv(
        cfg_pipe['metadata']['general'], usecols=['dicom_id', 'ViewPosition', 'PatientOrientationCodeSequence_CodeMeaning']
    )
    ap_ids = viewpos.loc[viewpos['ViewPosition'] == 'AP', 'dicom_id'].values.tolist()
    pos_ids = viewpos.loc[viewpos['PatientOrientationCodeSequence_CodeMeaning'] == 'Erect', 'dicom_id'].values.tolist()
    metadata = metadata[
        (~(metadata.dicom_id.isin(ap_ids) & (metadata.category_name == 'Cardiomegaly'))) &
        (metadata.dicom_id.isin(pos_ids)) &
        (metadata.category_name != 'Pneumonia')
    ]
    metadata = preprocess_meta(metadata)

    # Create raw priors and distributions if they do not exist
    path_ = cfg_pipe['priors']['savepath']
    if not force:
        try:
            distrs = import_distributions(path_)
        except (OSError, pickle.UnpicklingError, KeyError, TypeError):
            pass
    if cfg_pipe['priors']['from_raw']:
        priors = extract_priors(
            data=metadata,
            saveflag=True,
            savepath=path_
        )
    else:
        priors = import_priors(path_)
    create_distributions(
        priors=priors,
        saveflag=True,
        savepath=path_
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script that creates the prior distributions.')
    parser.add_argument('--config', required=False, type=str, default='./configs/pipeline.json',
        help='Path to the pipeline configuration file (pipeline.json).')
    parser.add_argument('--force', action='store_true',
        help='Regenerate priors even if distribution pickles already exist.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg_pipe = json.load(f)
    with open(cfg_pipe['mask_segment']['config_label'], 'r') as f:
        cfg_mask = json.load(f)
    main(force=args.force)

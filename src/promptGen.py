"""
Radiology prompt generation and anatomical restriction parsing for CXR editing.

Provides curated, frequency-weighted radiology report phrases per pathology label and maps
prompt text to spatial bounding regions used by :mod:`maskGen` to constrain mask placement.

Pipeline role:
    1. :func:`generate_prompt` samples a report-style phrase for a given pathology label
       (or a fixed negative finding string for ``'No Finding'``).
    2. :func:`get_restrictions` parses laterality (left/right/bilateral/any) and anatomical
       region keywords from the prompt text.
    3. Returned percentage-coordinate boxes (see :data:`restrictions_dict`) restrict where
       prior-based bounding boxes may be sampled inside the lung segmentation.

Consumers:
    - :mod:`inpaintGen` — calls :func:`generate_prompt` when ``label_as_prompt=False``.
    - :mod:`maskGen` — calls :func:`get_restrictions` in :meth:`Mask.generate_mask`.

Coordinate convention:
    Restriction boxes use percentage coordinates relative to the lung segmentation crop,
    formatted as ``((x_lo, y_lo), (x_hi, y_hi))``. Values may extend slightly outside
    ``[0, 100]`` (see CheXmask zone definitions). When ``invert_left=True``,
    :func:`get_from_dict` mirrors left-lung x bounds so priors fitted on right-lung data
    apply consistently to both sides (same convention as :mod:`maskGen`).

Usage:
    >>> from promptGen import generate_prompt, get_restrictions
    >>> prompt = generate_prompt('Atelectasis')
    >>> lung_side, boxes = get_restrictions(prompt)
"""

import numpy as np
import re
from typing import TypeAlias

# Percentage-coordinate restriction box: ((x_lo, y_lo), (x_hi, y_hi)).
BBox: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


# Frequency-weighted prompt phrases per pathology label (MS-CXR curated subset).
# Each entry: {'texts': list[str], 'probs': list[float]} with len(texts) == len(probs).
prompts_red_dict = {
    'Atelectasis': {
        'texts' : [
            'Bibasilar atelectasis.', 'Left basilar atelectasis.', 'Basilar atelectasis.', 
            'Bibasilar subsegmental atelectasis.', 'Right basilar atelectasis.', 'Left lower lobe atelectasis.', 
            'Atelectasis in the lung bases.', 'Left basilar subsegmental atelectasis.', 'Streaky bibasilar atelectasis.', 
            'Subsegmental atelectasis.', 'Linear bibasilar atelectasis.', 'Atelectasis.', 'Left lower lobe collapse.', 
            'Right lower lobe atelectasis.', 'Right basilar subsegmental atelectasis.', 'Patchy bibasilar atelectasis.', 
            'Right upper lobe collapse.', 'Right middle lobe collapse.'
        ],
        'probs': [
            0.6406, 0.1647, 0.038, 0.0341, 0.038, 0.018, 0.0106, 0.0053, 0.0042, 0.0042, 0.0063, 0.0158, 0.0032, 0.0021, 
            0.0042, 0.0063, 0.0022, 0.0022
        ]
    },
    'Cardiomegaly': {
        'texts' : [
            'Cardiomegaly.', 'Enlarged cardiac silhouette.', 'Enlargement of the cardiac silhouette.', 
            'Prominent cardiac silhouette.', 'Enlarged heart.'
        ],
        'probs': [0.7846, 0.194, 0.0154, 0.0018, 0.0042]
    },
    'Consolidation': {
        'texts': [
            'Left lower lobe consolidation.', 'Right lower lobe consolidation.', 
            'Patchy consolidation in the mid left lung.', 'Patchy consolidation in the right lung.', 
            'Patchy consolidation in the right lower lobe.', 'Left consolidation.', 
            'Patchy bilateral pulmonary consolidations.', 'Bilateral consolidations.', 'Right middle lobe consolidation.', 
            'Right upper lobe consolidation.'
        ],
        'probs': [0.3064, 0.2401, 0.0704, 0.0704, 0.1232, 0.0352, 0.0352, 0.0340, 0.0511, 0.0340]
    },
    'Edema': {
        'texts': [
            'Pulmonary edema.', 'Interstitial pulmonary edema.', 'Interstitial edema.', 'Edema.', 
            'Peribronchial cuffing consistent with pulmonary edema.'
        ],
        'probs': [0.731, 0.1333, 0.1023, 0.0175, 0.0159]
    },
    'Lung Opacity': {
        'texts': [
            'Right lower lobe infiltrate.', 'Right lower lobe opacity.', 'Left lower lobe opacity.', 
            'Bilateral lower lobe infiltrates.', 'Left lower lobe infiltrate.', 'Patchy bilateral pulmonary opacities.', 
            'Patchy left lower lobe opacity.', 'Bibasilar opacities.', 
            'Patchy ground-glass opacities at the right lung base.', 'Left basilar opacity.', 'Right basilar opacity.', 
            'Lower lung opacity.', 'Patchy ground-glass opacities in the left lower lung.', 
            'Patchy bibasilar opacities.'
        ],
        'probs': [
            0.1635, 0.1499, 0.1908, 0.0681, 0.0681, 0.0514, 0.0681, 0.0409, 0.0386, 0.0273, 0.0273, 0.0273, 0.0257, 0.053
        ]
    },
    'Pleural Effusion': {
        'texts': [
            'Bilateral pleural effusions.', 'Right pleural effusion.', 'Left pleural effusion.', 'Right effusion.', 
            'Bilateral effusions.', 'Right-sided pleural effusion.', 'Left-sided pleural effusion.', 'Left effusion.'
        ],
        'probs': [0.3979, 0.2429, 0.2351, 0.0388, 0.031, 0.0284, 0.0207, 0.0052]
    },
    'Pneumothorax': {
        'texts': [
            'Right apical pneumothorax.', 'Left apical pneumothorax.', 'Right pneumothorax.', 'Left pneumothorax.', 
            'Pneumothorax.', 'Apical pneumothorax.', 'Bilateral pneumothoraces.'
        ],
        'probs': [0.3472, 0.3208, 0.1774, 0.1245, 0.0151, 0.0075, 0.0075]
    }
}

# CheXmask anatomical zone bounds in percentage coordinates ((x_lo, y_lo), (x_hi, y_hi)).
restrictions_dict = {
    'left apical zone':         ((-6.01, 0.0), (98.66, 36.39)),
    'left hemidiaphragm+costophrenic angle': ((0.0, 66.8), (119.48, 108.35)),
    'left hilar structures':    ((-2.47, 20.89), (55.51, 73.5)),
    'left lower lung zone':     ((-10.13, 52.98), (100.0, 100.0)),
    'left lower lung lobe':     ((-10.13, 26.42), (106.55, 100.0)),
    'left mid lung zone':       ((-6.37, 26.42), (106.55, 70.15)),
    'left upper lung lobe':     ((-6.37, 1.01), (106.55, 70.15)),
    'left upper lung zone':     ((-2.54, 1.01), (104.16, 53.09)),
    'right apical zone':        ((4.39, 0.0), (101.6, 36.77)),
    'right hemidiaphragm+costophrenic angle': ((-19.1, 66.16), (131.24, 108.0)),
    'right hilar structures':   ((44.8, 20.41), (102.06, 74.3)),
    'right lower lung zone':    ((0.0, 52.67), (108.7, 100.0)),
    'right lower lung lobe':    ((0.0, 52.67), (108.7, 100.0)),
    'right mid lung zone':      ((-3.5, 25.54), (104.61, 71.31)),
    'right mid lung lobe':      ((-3.5, 25.54), (104.61, 71.31)),
    'right upper lung zone':    ((-1.51, 1.2), (100.0, 55.01)),
    'right upper lung lobe':    ((-1.51, 1.2), (100.0, 55.01))
}


def generate_prompt(label: str) -> str:
    """Sample a radiology report phrase for a pathology label.

    For ``'No Finding'``, returns a fixed negative finding string. For all other labels,
    draws one phrase from :data:`prompts_red_dict` using the configured probability weights.

    Args:
        label (str): Pathology label (key in :data:`prompts_red_dict`) or ``'No Finding'``.

    Returns:
        str: Sampled prompt text.

    Raises:
        KeyError: If ``label`` is not ``'No Finding'`` and not present in
            :data:`prompts_red_dict`.
    """
    if label == 'No Finding':
        return 'No acute cardiopulmonary process.'
    prompts_dict = prompts_red_dict
    return np.random.choice(prompts_dict[label]['texts'], size=1, p=prompts_dict[label]['probs']).item()


def get_from_dict(section: str, direction: str, invert_left: bool) -> list[BBox]:
    """Look up percentage-coordinate restriction boxes for an anatomical section.

    Args:
        section (str): Anatomical subsection name (e.g. ``'lower lung lobe'``,
            ``'apical zone'``). Prefixed with ``'left '`` or ``'right '`` when indexing
            :data:`restrictions_dict`.
        direction (str): Laterality filter — ``'left'``, ``'right'``, ``'both'``, or
            ``'any'``. Values other than ``'left'`` include the right-lung box; values
            other than ``'right'`` include the left-lung box.
        invert_left (bool): If True, mirror left-lung x coordinates (``|100 - x|``) so
            right-lung priors align with left-lung sampling in :mod:`maskGen`.

    Returns:
        list[BBox]: One or two restriction boxes depending on ``direction``.
    """
    restrictions = []
    if direction != 'left':
        restrictions.append(restrictions_dict[f'right {section}'])
    if direction != 'right':
        lrestriction = restrictions_dict[f'left {section}']
        if invert_left:
            lrestriction = (
                (abs(100 - lrestriction[1][0]), lrestriction[0][1]), (abs(100 - lrestriction[0][0]), lrestriction[1][1])
            )
        restrictions.append(lrestriction)
    return restrictions


def get_restrictions(prompt: str, invert_left: bool = True) -> tuple[str, list[BBox]]:
    """Parse laterality and anatomical region restrictions from prompt text.

    Uses keyword and regex matching on lowercased prompt text. Region keywords are
    checked in priority order (peribronchial → lobar → zone → basilar → apical).
    If no region matches, returns the full image bounds ``((0, 0), (100, 100))`` — e.g.
    ``'Bilateral pneumothoraces.'`` sets laterality to ``'both'`` but keeps full bounds
    because no regional keyword is present.

    Args:
        prompt (str): Radiology report phrase (from :func:`generate_prompt` or manual input).
        invert_left (bool): Passed through to :func:`get_from_dict` for left-lung x mirroring.
            Defaults to True (matches :mod:`maskGen` usage).

    Returns:
        tuple[str, list[BBox]]: ``(lung_side, boxes)`` where ``lung_side`` is one of
        ``'left'``, ``'right'``, ``'both'``, or ``'any'``, and ``boxes`` is a list of
        allowed center regions in percentage coordinates.
    """
    lung = 'any'
    prompt = prompt.lower()
    # Restrictions on x-axis
    if re.compile(r'(bibasilar|bilateral|lung bases)').search(prompt):
        lung = 'both'
    elif 'left' in prompt:
        lung = 'left'
    elif 'right' in prompt:
        lung = 'right'
    if 'peribronchial' in prompt:
        return lung, get_from_dict('hilar structures', lung, invert_left)
    # Restrictions on y-axis
    if 'lower lobe' in prompt:
        return lung, get_from_dict('lower lung lobe', lung, invert_left)
    elif 'lower lung' in prompt:
        return lung, get_from_dict('lower lung zone', lung, invert_left)
    elif 'upper lobe' in prompt:
        return lung, get_from_dict('upper lung lobe', lung, invert_left)
    elif 'upper lung' in prompt:
        return lung, get_from_dict('upper lung zone', lung, invert_left)
    elif re.compile(r'right[\w\s]*middle lobe').search(prompt):
        return lung, get_from_dict('mid lung lobe', lung, invert_left)
    elif re.compile(r'mid[dle]*\s').search(prompt):
        return lung, get_from_dict('mid lung zone', lung, invert_left)
    elif re.compile(r'(basilar|base)').search(prompt):
        return lung, get_from_dict('hemidiaphragm+costophrenic angle', lung, invert_left)
    elif 'apical' in prompt:
        return lung, get_from_dict('apical zone', lung, invert_left)
    return lung, [((0., 0.), (100., 100.))]
"""
Distribution Fitting and Constrained Sampling for CXR Priors.

Fits parametric distributions to normalized prior statistics (center positions and
width/height ratios) and provides sampling utilities used by :mod:`priorDistrGen` (fitting)
and :mod:`maskGen` (constrained sampling during mask generation).

Workflow:
    1. :func:`fit_multivar` / :func:`fit_univar` fit 1D distributions via ``distfit`` or
       2D multivariate normal / log-normal models to training data.
    2. Fitted models are wrapped in :class:`Distr1D` or :class:`Distr2D` subclasses.
    3. :func:`sample_jointdistr` and :func:`sample_distr` draw samples, optionally truncated
       to anatomical bounds, and return the probability mass of the discretization bin
       (``unit_equiv`` grid cell) for importance weighting.
"""

import numpy as np
import math
import pandas as pd

from matplotlib.pyplot import close as pltclose
from typing import Optional, Union
from distfit import distfit
from scipy import stats

from utils.MVNSampler import TruncatedMVN


class Distr1D:
    """Wrapper around a fitted 1D scipy-compatible distribution.

    Adds hyperrectangle (interval) CDF evaluation and inverse-CDF sampling truncated to
    a sub-interval, used when mask generation must respect anatomical bounds.

    Args:
        distr: A scipy ``rv_continuous`` instance (typically the best-fit model from
            ``distfit`` stored in ``result.summary.model.iloc[0]``).
    """

    def __init__(self, distr):
        self.distr = distr
        self.sampler = stats.uniform()

    def sample(self):
        """Draw an unconstrained sample from the fitted distribution.

        Returns:
            float: A single random variate.
        """
        return self.distr.rvs()

    def hyperrectangle_cdf(self, lb, ub):
        """Probability mass of the interval ``(lb, ub]``.

        Args:
            lb (float): Lower bound (exclusive in the truncated CDF convention used here).
            ub (float): Upper bound.

        Returns:
            float: ``CDF(ub) - CDF(lb)``.
        """
        return self.distr.cdf(ub) - self.distr.cdf(lb)

    def truncated_sample(self, lb, ub):
        """Sample from the distribution conditioned on lying in ``(lb, ub]``.

        Uses inverse transform sampling on the truncated CDF.

        Args:
            lb (float): Lower truncation bound.
            ub (float): Upper truncation bound.

        Returns:
            float: A single truncated sample.
        """
        u = self.sampler.rvs()
        cdf_lb, cdf_ub = self.distr.cdf([lb, ub])
        return self.distr.ppf(u * (cdf_ub - cdf_lb) + cdf_lb)

    def truncated_pdf(self, value, lb, ub):
        """Evaluate the truncated PDF at ``value`` given support ``(lb, ub]``.

        Args:
            value (float): Point at which to evaluate the density.
            lb (float): Lower truncation bound.
            ub (float): Upper truncation bound.

        Returns:
            float: Renormalized PDF value, or 0 if ``value`` is outside ``(lb, ub]``.
        """
        if lb < value <= ub:
            cdf_lb, cdf_ub = self.distr.cdf([lb, ub])
            pdf_value = self.distr.pdf(value)
            return pdf_value / (cdf_ub - cdf_lb)
        else:
            return 0

    def truncated_cdf(self, value, lb, ub):
        """Evaluate the truncated CDF at ``value`` relative to support ``(lb, ub]``.

        Args:
            value (float): Upper limit of the cumulative probability.
            lb (float): Lower truncation bound.
            ub (float): Upper truncation bound.

        Returns:
            float: Normalized cumulative probability in ``[0, 1]``.
        """
        cdf_value, cdf_lb, cdf_ub = self.distr.cdf([value, lb, ub])
        return (cdf_value - cdf_lb) / (cdf_ub - cdf_lb)

    def hyperrectangle_truncated_cdf(self, lb, ub, glb, gub):
        """Probability mass of ``(lb, ub]`` under the distribution truncated to ``(glb, gub]``.

        Args:
            lb (float): Lower bound of the query interval.
            ub (float): Upper bound of the query interval.
            glb (float): Global lower truncation bound.
            gub (float): Global upper truncation bound.

        Returns:
            float: Renormalized probability of the query interval.
        """
        return self.truncated_cdf(ub, glb, gub) - self.truncated_cdf(lb, glb, gub)


class Distr2D:
    """Base class for 2D distributions with axis-aligned hyperrectangle operations.

    Subclasses must implement :meth:`sample`, :meth:`pdf`, and :meth:`cdf`. Provides
    truncated PDF/CDF helpers for rectangular regions. Truncated sampling is implemented
    only in :class:`MVNormal` and :class:`MVLognormal`.

    Args:
        distr: Underlying scipy multivariate distribution (set by subclasses).
    """

    def __init__(self, distr):
        self.distr = distr

    def hyperrectangle_cdf(self, lb, ub):
        """Probability mass of the axis-aligned rectangle defined by corner points.

        Uses inclusion–exclusion on the joint CDF.

        Args:
            lb (array-like): Lower corner ``[lb_x, lb_y]``.
            ub (array-like): Upper corner ``[ub_x, ub_y]``.

        Returns:
            float: Probability mass of the rectangle.
        """
        return np.nansum([self.cdf(ub), self.cdf(lb), -self.cdf([lb[0], ub[1]]),
            -self.cdf([ub[0], lb[1]])])

    def truncated_sample(self, lb, ub):
        """Draw a sample truncated to the rectangle ``(lb, ub]``.

        Not implemented in the base class; see :class:`MVNormal` and :class:`MVLognormal`.
        """
        pass

    def truncated_pdf(self, value, lb, ub):
        """Evaluate the truncated joint PDF at ``value``.

        Args:
            value (array-like): Point ``[x, y]``.
            lb (array-like): Lower corner of the truncation rectangle.
            ub (array-like): Upper corner of the truncation rectangle.

        Returns:
            float: Renormalized PDF, or 0 outside the truncation region.
        """
        if (lb[0] < value[0] <= ub[0]) and (lb[1] < value[1] <= ub[1]):
            cdf_restr = self.hyperrectangle_cdf(lb, ub)
            pdf_value = self.pdf(value)
            return pdf_value / cdf_restr
        else:
            return 0

    def truncated_cdf(self, value, lb, ub):
        """Evaluate the truncated CDF over ``(lb, value]``.

        Args:
            value (array-like): Upper corner of the query rectangle.
            lb (array-like): Lower corner of the truncation rectangle.
            ub (array-like): Upper corner of the truncation rectangle.

        Returns:
            float: Normalized cumulative probability.
        """
        cdf_restr = self.hyperrectangle_cdf(lb, ub)
        cdf_rel_value = self.hyperrectangle_cdf(lb, value)
        return cdf_rel_value / cdf_restr

    def hyperrectangle_truncated_cdf(self, lb, ub, glb, gub):
        """Probability mass of rectangle ``(lb, ub]`` under truncation to ``(glb, gub]``.

        Args:
            lb (array-like): Lower corner of the query rectangle.
            ub (array-like): Upper corner of the query rectangle.
            glb (array-like): Lower corner of the global truncation rectangle.
            gub (array-like): Upper corner of the global truncation rectangle.

        Returns:
            float: Renormalized probability mass.
        """
        return self.truncated_cdf(ub, glb, gub) + self.truncated_cdf(lb, glb, gub) - \
            self.truncated_cdf([lb[0], ub[1]], glb, gub) - self.truncated_cdf([ub[0], lb[1]], glb, gub)


class MVNormal(Distr2D):
    """Bivariate normal distribution fit to observed (x, y) or (w, h) prior data.

    Truncated sampling delegates to :class:`utils.MVNSampler.TruncatedMVN`.

    Args:
        data (np.ndarray): Observations of shape ``(N, 2)``.
    """

    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.cov = np.cov(data, rowvar=0)
        self.distr = stats.multivariate_normal(self.mean, self.cov)

    def sample(self):
        """Draw an unconstrained sample from the fitted MVN.

        Returns:
            np.ndarray: Sample of shape ``(2,)``.
        """
        return self.distr.rvs()

    def pdf(self, value):
        """Evaluate the joint PDF at ``value``.

        Args:
            value (array-like): Point ``[x, y]``.

        Returns:
            float: PDF value.
        """
        return self.distr.pdf(value)

    def cdf(self, value):
        """Evaluate the joint CDF at ``value``.

        Args:
            value (array-like): Point ``[x, y]``.

        Returns:
            float: CDF value.
        """
        return self.distr.cdf(value)

    def truncated_sample(self, lb, ub):
        """Draw a single sample from the MVN truncated to the axis-aligned box.

        Args:
            lb (array-like): Lower bounds ``[lb_x, lb_y]``.
            ub (array-like): Upper bounds ``[ub_x, ub_y]``.

        Returns:
            np.ndarray: Truncated sample of shape ``(2,)``.
        """
        sampler = TruncatedMVN(self.mean, self.cov, lb, ub)
        return sampler.sample(1).reshape(-1)


class MVLognormal(Distr2D):
    """Bivariate log-normal distribution fit to observed (x, y) or (w, h) prior data.

    Fits a MVN in log-space and exponentiates samples. Truncated sampling is performed in
    log-space via :class:`utils.MVNSampler.TruncatedMVN`.

    Args:
        data (np.ndarray): Observations of shape ``(N, 2)`` (must be positive).
    """

    def __init__(self, data):
        logdata = np.log(np.clip(data, 10e-12, None))
        self.logmean = np.mean(logdata, axis=0)
        self.logcov = np.cov(logdata, rowvar=0)
        self.distr = stats.multivariate_normal(self.logmean, self.logcov)

    def sample(self):
        """Draw an unconstrained sample from the fitted log-normal.

        Returns:
            np.ndarray: Sample of shape ``(2,)``.
        """
        return np.exp(self.distr.rvs())

    def pdf(self, value):
        """Evaluate the joint log-normal PDF at ``value``.

        Args:
            value (array-like): Point ``[x, y]`` (must be positive).

        Returns:
            float: PDF value.
        """
        return self.distr.pdf(np.log(value)) / np.prod(value)

    def cdf(self, value):
        """Evaluate the joint log-normal CDF at ``value``.

        Args:
            value (array-like): Point ``[x, y]``.

        Returns:
            float: CDF value.
        """
        return self.distr.cdf(np.log(value))

    def truncated_sample(self, lb, ub):
        """Draw a single sample from the log-normal truncated to the axis-aligned box.

        Args:
            lb (array-like): Lower bounds ``[lb_x, lb_y]`` (must be positive).
            ub (array-like): Upper bounds ``[ub_x, ub_y]``.

        Returns:
            np.ndarray: Truncated sample of shape ``(2,)``.
        """
        sampler = TruncatedMVN(self.logmean, self.logcov, np.log(lb), np.log(ub))
        return np.exp(sampler.sample(1).reshape(-1))


def save_stats(result, savepath, n_top=5):
    """Persist distfit results as a PDF plot and summary CSV.

    Args:
        result: A fitted ``distfit`` result object.
        savepath (str): Base path (without extension) for output files.
        n_top (int): Number of top candidate distributions to show in the PDF plot.
            Defaults to 5.
    """
    fig, _ = result.plot(chart='pdf', n_top=n_top)
    fig.savefig(f'{savepath}_pdf.png', bbox_inches='tight')
    pltclose(fig)
    result.summary.to_csv(f'{savepath}_summary.csv', index=False)


def save_stats_2d(model, savepath):
    """Write a 2D fit summary CSV for MVNormal or MVLognormal (diagnostics only).

    Args:
        model (MVNormal | MVLognormal): Fitted 2D model.
        savepath (str): Base path (without extension) for the output CSV.
    """
    if isinstance(model, MVLognormal):
        mean = model.logmean
        cov = model.logcov
        model_name = 'MVLognormal'
    else:
        mean = model.mean
        cov = model.cov
        model_name = 'MVNormal'
    pd.DataFrame([{
        'model': model_name,
        'mean_x': mean[0],
        'mean_y': mean[1],
        'cov_xx': cov[0, 0],
        'cov_xy': cov[0, 1],
        'cov_yy': cov[1, 1],
    }]).to_csv(f'{savepath}_xy_summary.csv', index=False)


def test_distributions(data, distributions):
    """Fit candidate 1D distributions to data using ``distfit``.

    Args:
        data (np.ndarray): 1D array of observations.
        distributions (list[str]): Distribution family names accepted by ``distfit``.

    Returns:
        distfit: Fitted ``distfit`` object with ``summary`` ranking candidate models.
    """
    fitter = distfit(distr=distributions)
    fitter.fit_transform(data)
    return fitter


def fit_multivar(data, independent, distributions, savestats=False, savepath: str = '../results/priors/'):
    """Fit distributions to a bivariate prior dataset.

    When ``independent`` is True, fits separate 1D models to the ``x`` and ``y`` columns
    via ``distfit``. Otherwise fits a joint 2D model — :class:`MVLognormal` if
    ``distributions == 'MVLN'``, else :class:`MVNormal`.

    Args:
        data (dict): Mapping with exactly two keys (e.g. ``{'x': [...], 'y': [...]}``).
        independent (bool): If True, fit marginals independently; if False, fit jointly.
        distributions (list[str] | str): For 1D fitting, a list of distfit family names.
            For 2D fitting, ``'MVLN'`` selects log-normal; any other value selects MVN.
        savestats (bool): Whether to write fit plots and CSV summaries.
            Defaults to False.
        savepath (str): Base output path prefix. Defaults to ``'../results/priors/'``.

    Returns:
        dict: For independent fits, ``{'x': Distr1D, 'y': Distr1D}``. For joint fits,
            ``{'xy': MVNormal | MVLognormal}``.
    """
    fits = {}
    x = np.array(data[list(data.keys())[0]])
    y = np.array(data[list(data.keys())[1]])
    if independent:
        for k, v in [['x', x], ['y', y]]:
            result = test_distributions(v, distributions)
            if savestats:
                save_stats(result, f'{savepath}_{k}', n_top=5)
            fits[k] = Distr1D(result.summary.model.iloc[0])
    else:
        xy = np.column_stack([x, y])
        fits['xy'] = MVLognormal(xy) if distributions == 'MVLN' else MVNormal(xy)
        if savestats:
            save_stats_2d(fits['xy'], savepath)
    return fits


def fit_univar(data, distributions, savestats=False, savepath: str = '../results/priors/'):
    """Fit a 1D distribution to a univariate prior dataset.

    Args:
        data (dict): Mapping with a single key (e.g. ``{'x': [...]}``).
        distributions (list[str]): Distribution family names for ``distfit``.
        savestats (bool): Whether to write fit artifacts. Defaults to False.
        savepath (str): Base output path prefix. Defaults to ``'../results/priors/'``.

    Returns:
        dict: ``{'x': Distr1D}`` wrapping the best-fit model.
    """
    fits = {}
    x = np.array(data[list(data.keys())[0]])
    result = test_distributions(x, distributions)
    if savestats:
        save_stats(result, f'{savepath}_x', n_top=5)
    fits['x'] = Distr1D(result.summary.model.iloc[0])
    return fits


def sample_jointdistr(distrs, bounds: Optional[Union[list, tuple]] = None, unit_equiv: list = [1, 1]):
    """Sample from a bivariate prior and return the discretization-bin probability.

    Handles both joint 2D models (``distrs`` has key ``'xy'``) and independent 1D marginals
    (keys ``'x'`` and ``'y'``). When ``bounds`` is provided, samples are drawn from the
    truncated distribution. The returned probability is the mass of the ``unit_equiv``-sized
    grid cell containing the sample, used for importance weighting in ``maskGen``.

    Args:
        distrs (dict): Output of :func:`fit_multivar` — either ``{'xy': Distr2D}`` or
            ``{'x': Distr1D, 'y': Distr1D}``.
        bounds (tuple, optional): ``(lb, ub)`` truncation bounds, each an array-like of
            length 2. If None, samples are unconstrained.
        unit_equiv (list): Grid cell width per dimension for probability evaluation.
            Defaults to ``[1, 1]`` (percentage units).

    Returns:
        tuple[np.ndarray, float]: Sample ``[x, y]`` and the hyperrectangle (truncated)
            probability mass of its containing bin.
    """
    if len(distrs) == 1:
        if bounds is None:
            sample = distrs['xy'].sample()
            prob = distrs['xy'].hyperrectangle_cdf(
                (sample // unit_equiv) * unit_equiv, (sample // unit_equiv + 1) * unit_equiv
            )
        else:
            sample = distrs['xy'].truncated_sample(*bounds)
            prob = distrs['xy'].hyperrectangle_truncated_cdf(
                (sample // unit_equiv) * unit_equiv, (sample // unit_equiv + 1) * unit_equiv, *bounds
            )
    else:
        sample = np.zeros(2)
        prob = 1
        for dim_idx, dim_name in zip(range(2), ['x', 'y']):
            sample_, prob_ = sample_distr(
                distrs[dim_name],
                bounds if bounds is None else (bounds[0][dim_idx], bounds[1][dim_idx]),
                unit_equiv[dim_idx]
            )
            sample[dim_idx] = sample_
            prob *= prob_
    return sample, prob


def sample_distr(distr, bounds: Optional[Union[list, tuple]] = None, unit_equiv: float = 1):
    """Sample from a 1D prior and return the discretization-bin probability.

    Args:
        distr (Distr1D): Fitted 1D distribution wrapper.
        bounds (tuple, optional): ``(lb, ub)`` truncation bounds. If None, unconstrained.
        unit_equiv (float): Grid cell width for probability evaluation. Defaults to 1.

    Returns:
        tuple[float, float]: Sample value and the hyperrectangle (truncated) probability
            mass of its containing bin.
    """
    if bounds is None:
        sample = distr.sample()
        prob = distr.hyperrectangle_cdf(
            (sample // unit_equiv) * unit_equiv, (sample // unit_equiv + 1) * unit_equiv
        )
    else:
        sample = distr.truncated_sample(*bounds)
        prob = distr.hyperrectangle_truncated_cdf(
            (sample // unit_equiv) * unit_equiv, (sample // unit_equiv + 1) * unit_equiv, *bounds
        )
    return sample, prob

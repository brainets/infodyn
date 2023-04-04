"""Connectivity using Interaction Information"""
import numpy as np
import xarray as xr

from sklearn.decomposition import PCA

from frites.conn import conn_io
from frites.core import mi_nd_gg, mi_model_nd_gd, copnorm_nd
from frites.io import set_log_level, logger, check_attrs
from frites.config import CONFIG

from mne.utils import ProgressBar


def _mi_estimation(x, y, mi_type):
    """Compute the MI on each roi.

    x.shape = (n_times, {1, Nd}, n_trials)
    y.shape = ({1, Nd}, n_trials)
    """
    x = np.ascontiguousarray(x)
    cfg_mi = CONFIG["KW_GCMI"]
    if mi_type == 'cc':
        y = np.atleast_2d(y)[np.newaxis, ...]
        # repeat y to match x shape
        y = np.tile(y, (x.shape[0], 1, 1))
        return mi_nd_gg(x, y, **cfg_mi)
    elif mi_type == 'cd':
        return mi_model_nd_gd(x, y, **cfg_mi)


def conn_ii(data, y, roi=None, times=None, mi_type='cc', gcrn=True, dt=1,
             sfreq=None, verbose=None, **kw_links):
    """Interaction Information on connectivity pairs and behavioral variable.

    This function can be used to untangle how the information about a stimulus
    is carried inside a brain network.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    y : array_like
        The feature of shape (n_trials,). This feature vector can either be
        categorical and in that case, the mutual information type has to 'cd'
        or y can also be a continuous regressor and in that case the mutual
        information type has to be 'cc'
    roi : array_like | None
        Array of region of interest name of shape (n_roi,)
    times : array_like | None
        Array of time points of shape (n_times,)
    mi_type : {'cc', 'cd'}
        Mutual information type. Switch between :
            * 'cc' : if the y input is a continuous regressor
            * 'cd' : if the y input is a discret vector with categorical
              integers inside
    gcrn : bool | True
        Specify if the Gaussian Copula Rank Normalization should be applied.
        Default is True.
    kw_links : dict | {}
        Additional arguments for selecting links to compute are passed to the
        function :func:`frites.conn.conn_links`

    Returns
    -------
    mi_node : array_like
        The array of mutual infromation estimated on each node of shape
        (n_roi, n_times)
    interinfo : array_like
        The interaction information in the network of shape (n_pairs, n_times)

    See also
    --------
    conn_links
    """
    # _________________________________ INPUTS ________________________________
    # inputs conversion
    kw_links.update({'directed': False, 'net': False})
    data, cfg = conn_io(
        data, y=y, times=times, roi=roi, agg_ch=False, win_sample=None,
        name='II', sfreq=sfreq, verbose=verbose, kw_links=kw_links
    )

    # extract variables
    x, attrs = data.data, cfg['attrs']
    y, roi, times = data['y'].data, data['roi'].data, data['times'].data
    x_s, x_t = cfg['x_s'], cfg['x_t']
    roi_p, n_pairs = cfg['roi_p'], len(x_s)

    assert dt >= 1
    # build the indices when using multi-variate mi
    idx = np.mgrid[0:len(times) - dt + 1, 0:dt].sum(0)
    times = times[idx].mean(1)
    n_trials, n_roi, n_times = len(y), len(roi), len(times)

    logger.info(f"Compute II on {n_pairs} connectivity pairs")
    # gcrn
    if gcrn:
        logger.info("    Apply the Gaussian Copula Rank Normalization")
        x = copnorm_nd(x, axis=0)
        if mi_type == 'cc':
            y = copnorm_nd(y, axis=0)

    # get the mi function to use
    fcn = {'cc': mi_nd_gg, 'cd': mi_model_nd_gd}[mi_type]

    # transpose the data to be (n_roi, n_times, 1, n_trials)
    x = np.transpose(x, (1, 2, 0))

    # __________________________________ PID __________________________________
    # compute mi on each node of the network
    logger.info("    Computing Interaction Information for all pairs in the network")
    pbar = ProgressBar(range(2 * n_roi + n_pairs),
                       mesg='Estimating MI on each node I(X;S)')
    mi_node = np.zeros((n_roi, n_times), dtype=float)
    for n_r in range(n_roi):
        mi_node[n_r, :] = _mi_estimation(x[n_r, idx, :], y, mi_type)
        pbar.update_with_increment_value(1)

    pbar._tqdm.desc = 'Estimating total information I(X,Y;S)'
    infotot = np.zeros((n_pairs, n_times))
    for n_p, (s, t) in enumerate(zip(x_s, x_t)):
        _x_s, _x_t = x[s, ...], x[t, ...]

        # total information estimation
        x_st = np.concatenate((_x_s[idx, ...], _x_t[idx, ...]), axis=1)
        infotot[n_p, :] = _mi_estimation(x_st, y, mi_type)

        pbar.update_with_increment_value(1)

    # interaction information
    interinfo = infotot - mi_node[x_s, :] - mi_node[x_t, :]

    # _______________________________ OUTPUTS _________________________________
    kw = dict(dims=('roi', 'times'), coords=(roi, times),
              attrs=check_attrs(attrs))
    kw_pairs = dict(dims=('roi', 'times'), coords=(roi_p, times))
    mi_node = xr.DataArray(mi_node, name='mi_node', **kw)
    interinfo = xr.DataArray(interinfo, name='interinfo', **kw_pairs)

    return mi_node, interinfo


if __name__ == '__main__':
    from frites.simulations import StimSpecAR
    import matplotlib.pyplot as plt

    ar_type = 'hga'
    n_stim = 2
    n_epochs = 400

    # Simulate an AR model stimulus-specific HGA with redundancy
    # larger than synergy, that is negative Interaction Information
    ss = StimSpecAR()
    ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim)

    plt.figure(figsize=(12, 6))

    mi_node, interinfo = conn_ii(
        ar, 'trials', roi='roi', times='times', mi_type='cd', dt=1,
        verbose=False)

    times = interinfo['times'].data

    plt.plot(times, interinfo.squeeze(), color='blue', linestyle='--',
             label='Interaction Information')
    plt.legend()

    plt.show()

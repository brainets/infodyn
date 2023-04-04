"""Connectivity using Partial Information Decomposition."""
import numpy as np
import xarray as xr
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


def conn_pid(data, y, roi=None, times=None, mi_type='cc', gcrn=True, dt=1,
             sfreq=None, verbose=None, **kw_links):
    """Compute the Partial Information Decomposition on connectivity pairs.

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
    unique : array_like
        The unique contribution of each node of shape (n_roi, n_times)
    infotot : array_like
        The total information in the network of shape (n_pairs, n_times)
    redundancy : array_like
        The redundancy in the network of shape (n_pairs, n_times)
    synergy : array_like
        The synergy in the network of shape (n_pairs, n_times)

    See also
    --------
    conn_links
    """
    # _________________________________ INPUTS ________________________________
    # inputs conversion
    kw_links.update({'directed': False, 'net': False})
    data, cfg = conn_io(
        data, y=y, times=times, roi=roi, agg_ch=False, win_sample=None,
        name='PID', sfreq=sfreq, verbose=verbose, kw_links=kw_links
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

    logger.info(f"Compute PID on {n_pairs} connectivity pairs")
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
    logger.info("    Estimating PID in the network")
    pbar = ProgressBar(range(2 * n_roi + n_pairs),
                       mesg='Estimating MI on each node')
    mi_node = np.zeros((n_roi, n_times), dtype=float)
    for n_r in range(n_roi):
        mi_node[n_r, :] = _mi_estimation(x[n_r, idx, :], y, mi_type)
        pbar.update_with_increment_value(1)

    pbar._tqdm.desc = 'Estimating total information and redundancy'
    infotot = np.zeros((n_pairs, n_times))
    redundancy = np.zeros((n_pairs, n_times))
    for n_p, (s, t) in enumerate(zip(x_s, x_t)):
        _x_s, _x_t = x[s, ...], x[t, ...]

        # total information estimation
        x_st = np.concatenate((_x_s[idx, ...], _x_t[idx, ...]), axis=1)
        infotot[n_p, :] = _mi_estimation(x_st, y, mi_type)

        # redundancy estimation
        redundancy[n_p, :] = np.c_[mi_node[s, :], mi_node[t, :]].min(1)

        pbar.update_with_increment_value(1)


    # estimate the unique information
    pbar._tqdm.desc = 'Estimating unique information and synergy'
    unique = np.zeros((n_roi, n_times))
    for n_r in range(n_roi):
        idx_red = np.logical_or(x_s == n_r, x_t == n_r)
        if not np.any(idx_red):  # some pairs might be absent
            continue
        red_all = redundancy[idx_red, :].min(0)
        unique[n_r, :] = mi_node[n_r, :] - red_all
        pbar.update_with_increment_value(1)

    # feature specific synergy
    synergy = infotot - mi_node[x_s, :] - mi_node[x_t, :] + redundancy

    # _______________________________ OUTPUTS _________________________________
    kw = dict(dims=('roi', 'times'), coords=(roi, times),
              attrs=check_attrs(attrs))
    kw_pairs = dict(dims=('roi', 'times'), coords=(roi_p, times))
    mi_node = xr.DataArray(mi_node, name='mi_node', **kw)
    unique = xr.DataArray(unique, name='Unique', **kw)
    infotot = xr.DataArray(infotot, name='Infotot', **kw_pairs)
    redundancy = xr.DataArray(redundancy, name='Redundancy', **kw_pairs)
    synergy = xr.DataArray(synergy, name='Synergy', **kw_pairs)

    return mi_node, unique, infotot, redundancy, synergy


if __name__ == '__main__':
    
    from frites.simulations import StimSpecAR
    import matplotlib.pyplot as plt

    ar_type = 'hga'
    n_stim = 2
    n_epochs = 500

    ss = StimSpecAR()
    ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim)

    plt.figure(figsize=(7, 8))
    ss.plot(cmap='bwr')
    plt.tight_layout()
    plt.show()

    mi_node, unique, infotot, redundancy, synergy = conn_pid(
        ar, 'trials', roi='roi', times='times', mi_type='cd', dt=1,
        verbose=False)

    plt.figure(figsize=(6, 6))

    times = unique['times'].data

    plt.plot(times, infotot.squeeze(), color='blue', linestyle='--',
             label=r"$I_{TOT}$")
    plt.plot(times, redundancy.squeeze(), color='red', label='Red')
    plt.plot(times, unique.sel(roi='x').squeeze(), color='orange',
             label=r"$Unique_{X}$")
    plt.plot(times, unique.sel(roi='y').squeeze(), color='purple',
             label=r"$Unique_{Y}$")
    plt.plot(times, synergy.squeeze(), color='green', label="Syn")
    plt.legend()

    plt.title('PID of task-related comodulation between neural times series')
    plt.xlim(-0.1, 0.5)
    plt.grid()

    plt.show()

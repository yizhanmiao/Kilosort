"""Cluster IDs can have gaps when a prior unit has no spikes in a new session.

Every per-unit output must then be indexed by unit ID, with `n_units` entries.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import torch

from kilosort import CCG
from kilosort.plots import plot_spike_positions
from kilosort.postprocessing import make_pc_features


def _ops():
    return {'iU': torch.tensor([0, 1, 2, 3]),
            'iCC': torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            'xc': np.zeros(4), 'yc': np.arange(4) * 20., 'dmin': 20,
            'dminx': 32, 'nearest_chans': 2}


@pytest.mark.parametrize('clu, n_units, expected', [
    ([0, 0, 2, 2], None, 3),     # empty unit in the middle
    ([0, 0, 1, 1], 4, 4),        # trailing empty units
    ([0, 0, 1, 1], None, 2),     # contiguous, unchanged
    ])
def test_make_pc_features_empty_units(clu, n_units, expected):
    clu = np.array(clu, 'int32')
    tF = torch.randn(4, 2, 6)
    kwargs = {} if n_units is None else {'n_units': n_units}
    _, feature_ind = make_pc_features(_ops(), clu, clu, tF.clone(), **kwargs)
    assert feature_ind.shape[0] == expected


@pytest.mark.parametrize('clu, n_units, expected', [
    ([0, 0, 2, 2], None, 3),
    ([0, 0, 1, 1], 4, 4),
    ([0, 0, 1, 1], None, 2),
    ])
def test_refract_n_units(clu, n_units, expected):
    clu = np.array(clu, 'int32')
    is_ref, R = CCG.refract(clu, np.array([0., 1, 2, 3]), n_units=n_units)
    assert len(is_ref) == len(R) == expected


def test_refract_n_units_does_not_change_contiguous_values():
    rng = np.random.default_rng(0)
    clu = np.repeat(np.arange(3), 200).astype('int32')
    st = np.sort(rng.uniform(0, 100, 600))
    a = CCG.refract(clu, st)
    b = CCG.refract(clu, st, n_units=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize('clu, n_units', [([0, 0, 2, 2], 3), ([0, 0, 1, 1], 4)])
def test_plot_spike_positions_empty_units(tmp_path, clu, n_units):
    clu = np.array(clu, 'int32')
    np.save(tmp_path / 'spike_positions.npy', np.random.rand(len(clu), 2))
    is_ref = np.ones(n_units)
    is_ref[-1] = 0   # last unit (possibly empty) is not refractory
    plot_spike_positions(clu, is_ref, tmp_path)
    assert (tmp_path / 'spike_positions.png').exists() or any(tmp_path.glob('*.png'))


@pytest.mark.parametrize('clu', [[0, 0, 2, 2], [0, 0, 1, 1]])
def test_save_to_phy_with_empty_units(tmp_path, clu):
    from kilosort.io import save_to_phy
    nt, n_pcs, n_units = 61, 2, 4     # units not in `clu` have no spikes
    ops = _ops()
    ops.update({
        'iCC_mask': torch.ones((2, 4)), 'Wrot': torch.eye(4), 'nt': nt,
        'fs': 30000., 'duplicate_spike_bins': 15,
        'wPCA': torch.randn(n_pcs, nt),
        'settings': {'acg_threshold': 0.2, 'ccg_threshold': 0.25,
                     'n_chan_bin': 4, 'fs': 30000., 'filename': 'x.bin'},
        })
    probe = {'chanMap': np.arange(4), 'xc': ops['xc'], 'yc': ops['yc'],
             'kcoords': np.zeros(4)}
    clu = np.array(clu, 'int32')
    st = np.zeros((4, 3))
    st[:, 0] = np.arange(4) * 1000
    st[:, 1] = clu
    out_dir, similar, is_ref, contam, kept = save_to_phy(
        st, clu, torch.randn(4, 2, n_pcs), torch.randn(n_units, 4, n_pcs),
        probe, ops, 0, results_dir=tmp_path)

    assert similar.shape == (n_units, n_units)
    assert len(is_ref) == len(contam) == n_units
    assert np.load(tmp_path / 'templates.npy').shape[0] == n_units
    assert np.load(tmp_path / 'pc_feature_ind.npy').shape[0] == n_units
    for name in ['ContamPct', 'Amplitude', 'KSLabel', 'group']:
        rows = (tmp_path / f'cluster_{name}.tsv').read_text().splitlines()
        assert len(rows) == n_units + 1

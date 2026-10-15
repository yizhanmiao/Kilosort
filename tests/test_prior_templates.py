import importlib
from pathlib import Path

import numpy as np
import torch

# `kilosort.run_kilosort` as an attribute is the function (kilosort/__init__.py
# re-exports it over the submodule), so fetch the module explicitly.
rk = importlib.import_module('kilosort.run_kilosort')


def _ops(n_templates):
    # nC=2 nearest chans for each of 4 chans; template j peaks on chan j % 4
    return {'iCC': torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            'iU': torch.arange(n_templates) % 4}


def _st(templates):
    st = np.zeros((len(templates), 3))
    st[:, 0] = np.arange(len(templates)) * 100
    st[:, 1] = templates
    return st


def test_cluster_with_prior_keeps_ids_and_offsets_new(monkeypatch):
    prior = torch.ones((2, 4, 2))
    st = _st([0, 2, 1, 2])          # templates 0,1 are prior; 2 is a new candidate
    tF = torch.ones((4, 2, 2))

    def fake_cluster_spikes(st_sub, tF_sub, ops, device, bfile, **kw):
        assert len(st_sub) == 2
        return np.array([0, 0]), torch.full((1, 4, 2), 5.), st_sub, tF_sub
    monkeypatch.setattr(rk, 'cluster_spikes', fake_cluster_spikes)

    clu, Wall, st_out, tF_out = rk.cluster_spikes_with_prior(
        st, tF, _ops(3), prior, torch.device('cpu'), bfile=None)

    assert list(clu) == [0, 2, 1, 2]
    assert Wall.shape == (3, 4, 2)
    assert torch.all(Wall[2] == 5.)
    assert len(st_out) == len(tF_out) == 4


def test_cluster_with_prior_drops_unclusterable_new_spikes(monkeypatch):
    prior = torch.ones((2, 4, 2))
    st = _st([0, 2, 1])
    tF = torch.ones((3, 2, 2))

    def failing_cluster_spikes(*a, **kw):
        raise ValueError('Wall is empty after `clustering_qr.run`')
    monkeypatch.setattr(rk, 'cluster_spikes', failing_cluster_spikes)

    clu, Wall, st_out, tF_out = rk.cluster_spikes_with_prior(
        st, tF, _ops(3), prior, torch.device('cpu'), bfile=None)

    assert list(clu) == [0, 1]
    assert Wall.shape == (2, 4, 2)
    assert len(st_out) == len(tF_out) == 2


def test_prior_dir_for_shank():
    assert rk._prior_dir_for_shank(None, 1) is None
    assert rk._prior_dir_for_shank('/a/b', None) == Path('/a/b')
    assert rk._prior_dir_for_shank('/a/b', 2) == Path('/a/b/shank_2')

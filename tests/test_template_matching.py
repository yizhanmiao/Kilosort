import numpy as np
import torch

from kilosort import template_matching


def _orthonormal_wpca(n_pcs=6, nt=61, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((nt, n_pcs)))
    return torch.from_numpy(q.T.astype('float32'))


def test_drop_prior_duplicates():
    ops = {'wPCA': _orthonormal_wpca(), 'nt': 61}
    rng = np.random.default_rng(1)
    prior = torch.zeros((2, 4, 6))
    prior[:, :2] = torch.from_numpy(rng.standard_normal((2, 2, 6)).astype('float32'))
    dup = 3 * prior[0]                                  # same shape, scaled
    other = torch.zeros((4, 6))
    other[2:] = torch.from_numpy(rng.standard_normal((2, 6)).astype('float32'))  # disjoint chans
    cand = torch.stack([dup, other])

    kept = template_matching.drop_prior_duplicates(prior, cand, ops)

    assert kept.shape == (1, 4, 6)
    assert torch.allclose(kept[0], other)


def test_drop_prior_duplicates_no_candidates():
    ops = {'wPCA': _orthonormal_wpca(), 'nt': 61}
    kept = template_matching.drop_prior_duplicates(
        torch.ones((2, 4, 6)), torch.zeros((0, 4, 6)), ops)
    assert kept.shape == (0, 4, 6)


def test_reestimate_keeps_unit_without_spikes():
    prior = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    iCC = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])   # nC=2 nearest chans per chan
    iU = torch.tensor([0, 2])
    clu = np.array([0, 0, -1])                          # unit 1 has no spikes
    tF = torch.tensor([[[1., 1.], [2., 2.]],
                       [[3., 3.], [4., 4.]],
                       [[9., 9.], [9., 9.]]])

    W = template_matching.reestimate_prior_templates(prior, clu, tF, iCC, iU)

    assert torch.allclose(W[0, 0], torch.tensor([2., 2.]))   # mean on chan 0
    assert torch.allclose(W[0, 1], torch.tensor([3., 3.]))   # mean on chan 1
    assert torch.all(W[0, 2:] == 0)
    assert torch.equal(W[1], prior[1])

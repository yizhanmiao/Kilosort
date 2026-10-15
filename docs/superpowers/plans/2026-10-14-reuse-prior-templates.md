# Reuse Prior Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `run_kilosort(..., previous_results_dir=...)` keep unit IDs from a prior sort, still find new units, and save a normal results folder that the next session can chain from.

**Architecture:**
1. Load the prior templates (`Wall`) and the prior basis (`wPCA`, `wTEMP`, whitening matrix) from the prior results folder, and force the current run to use that basis.
2. Run detection and first clustering as normal to get candidate templates.
3. Drop candidates that duplicate a prior template.
4. Match `cat(prior, candidates)` in one pass.
5. Spikes on prior templates keep the prior ID; the rest go through the existing final clustering, with IDs offset by `N`.
6. Re-estimate prior templates from this session's spikes.

**Tech Stack:** Python, PyTorch, NumPy, pytest (existing Kilosort4 code base).

**Spec:** `docs/superpowers/specs/2026-10-14-reuse-prior-templates-design.md`

## Global Constraints

- When `previous_results_dir=None`, behavior is unchanged. The existing tests must still pass.
- Prior identity comes from raw `templates.npy` (or `Wall.npy` if present). Phy curation is ignored.
- Unit IDs `0 … N-1` are the prior units. New units are numbered from `N` upward.
- A prior unit with no spikes this session keeps its old template row.
- The candidate-vs-prior duplicate cutoff is similarity `> 0.9`, scored with `CCG.similarity`.
- If the channel map, `fs`, `nt` or `n_pcs` differs from the prior run → `ValueError`, naming the setting and showing both values.
- If `ops.npy` or `templates.npy` is missing → `FileNotFoundError` naming the file.
- Run tests with `pytest`. Slow tests need `--runslow` and download data on first use. Add `--gpu` to use CUDA.

## Review Focus

1. **Prior unit with zero spikes this session:** it keeps its ID and old template, and saving to Phy doesn't crash. Pinned in Task 2 (`test_reestimate_keeps_unit_without_spikes`). `CCG.refract` already skips clusters with ≤10 spikes.
2. **Very few unmatched spikes:** `cluster_spikes` raises `ValueError` ("Wall is empty" / no suitable channels). The run should drop those spikes with a warning, not crash. Pinned in Task 3 (`test_cluster_with_prior_drops_unclusterable_new_spikes`).
3. **Different `bad_channels` between runs:** the channel map differs, so the run fails clearly before sorting. Pinned in Task 1 (`test_load_prior_templates_channel_mismatch`).
4. **Built-in `wPCA` is not exactly orthonormal:** the reconstruction uses a pseudo-inverse, so the round trip is still exact. Pinned in Task 1 (`test_load_prior_templates_round_trip`, which uses a non-orthonormal basis).
5. **Multi-shank runs (`shank_idx`):** the prior folder for shank *i* is `previous_results_dir/shank_i`, mirroring `set_files`. Pinned in Task 3 (`test_prior_dir_for_shank`).

---

## File Structure

- `kilosort/io.py`: add `load_prior_templates(results_dir, ops, device)`, which loads, validates and reconstructs `Wall`.
- `kilosort/template_matching.py`: add `drop_prior_duplicates(...)` and `reestimate_prior_templates(...)`.
- `kilosort/spikedetect.py`: in `run`, skip re-learning `wPCA`/`wTEMP` when `ops['prior_basis']` is set.
- `kilosort/run_kilosort.py`:
  - `run_kilosort` / `_sort`: new `previous_results_dir` argument.
  - `compute_preprocessing`: optional `whiten_mat`.
  - `detect_spikes`: optional `prior_Wall`.
  - New `cluster_spikes_with_prior(...)` and `_prior_dir_for_shank(...)`.
- Tests:
  - `tests/test_io.py` (extend)
  - `tests/test_template_matching.py` (new)
  - `tests/test_prior_templates.py` (new)
  - `tests/test_full_pipeline.py` (extend, slow)

---

### Task 1: Load and validate prior templates

**Files:**
- Modify: `kilosort/io.py` (add after `load_ops`, around line 527)
- Test: `tests/test_io.py` (append)

**Interfaces:**
- Produces: `io.load_prior_templates(results_dir, ops, device) -> dict` with keys:
  - `'Wall'`: float32 tensor on `device`, shape `(N, Nchan, n_pcs)`
  - `'wPCA'`: `(n_pcs, nt)`
  - `'wTEMP'`
  - `'whiten_mat'`: `(Nchan, Nchan)`

  `ops` must contain `ops['probe']['chanMap']`, `ops['fs']`, `ops['nt']` and `ops['settings']['n_pcs']`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_io.py`)

```python
def _write_prior(results_dir, Wall, wPCA, chan_map, fs=30000, nt=61, n_pcs=6):
    """Write a minimal ops.npy + templates.npy pair like save_to_phy/save_ops do."""
    prior_ops = {
        'fs': fs, 'nt': nt, 'settings': {'n_pcs': n_pcs},
        'probe': {'chanMap': chan_map},
        'wPCA': wPCA, 'wTEMP': wPCA.copy(), 'is_tensor': ['wPCA', 'wTEMP'],
        'preprocessing': {'whiten_mat': np.eye(len(chan_map), dtype='float32')},
    }
    np.save(results_dir / 'ops.npy', np.array(prior_ops))
    # Same formula as io.save_to_phy: (N, chan, pc) x (pc, t) -> (N, t, chan)
    templates = (Wall[..., np.newaxis] * wPCA).sum(axis=-2).transpose(0, 2, 1)
    np.save(results_dir / 'templates.npy', templates)


def _current_ops(chan_map, fs=30000, nt=61, n_pcs=6):
    return {'fs': fs, 'nt': nt, 'settings': {'n_pcs': n_pcs},
            'probe': {'chanMap': chan_map}}


def test_load_prior_templates_round_trip():
    rng = np.random.default_rng(0)
    Wall = rng.standard_normal((3, 4, 6)).astype('float32')
    wPCA = rng.standard_normal((6, 61)).astype('float32')  # not orthonormal
    chan_map = np.arange(4)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_prior(d, Wall, wPCA, chan_map)
        prior = io.load_prior_templates(d, _current_ops(chan_map), torch.device('cpu'))
    assert prior['Wall'].shape == (3, 4, 6)
    assert torch.allclose(prior['Wall'], torch.from_numpy(Wall), atol=1e-3)
    assert prior['whiten_mat'].shape == (4, 4)
    assert prior['wPCA'].shape == (6, 61)


def test_load_prior_templates_prefers_wall_npy():
    rng = np.random.default_rng(1)
    Wall = rng.standard_normal((2, 4, 6)).astype('float32')
    wPCA = rng.standard_normal((6, 61)).astype('float32')
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_prior(d, Wall, wPCA, np.arange(4))
        np.save(d / 'Wall.npy', Wall * 2)
        prior = io.load_prior_templates(d, _current_ops(np.arange(4)), torch.device('cpu'))
    assert torch.allclose(prior['Wall'], torch.from_numpy(Wall * 2))


def test_load_prior_templates_channel_mismatch():
    rng = np.random.default_rng(2)
    Wall = rng.standard_normal((2, 4, 6)).astype('float32')
    wPCA = rng.standard_normal((6, 61)).astype('float32')
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_prior(d, Wall, wPCA, np.arange(4))
        with pytest.raises(ValueError, match='channel map'):
            io.load_prior_templates(d, _current_ops(np.array([0, 1, 3])), torch.device('cpu'))
        with pytest.raises(ValueError, match='fs'):
            io.load_prior_templates(d, _current_ops(np.arange(4), fs=25000), torch.device('cpu'))


def test_load_prior_templates_missing_file():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError, match='ops.npy'):
            io.load_prior_templates(Path(d), _current_ops(np.arange(4)), torch.device('cpu'))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_io.py -k prior -v`
Expected: 4 FAIL with `AttributeError: module 'kilosort.io' has no attribute 'load_prior_templates'`

- [ ] **Step 3: Implement** (in `kilosort/io.py`, directly after `load_ops`)

```python
def load_prior_templates(results_dir, ops, device):
    """Load templates and the basis they live in from a previous sorting result.

    Parameters
    ----------
    results_dir : str or pathlib.Path
        Results folder of a previous `run_kilosort` call.
    ops : dict
        Current run's ops, used to check that both runs are compatible.
    device : torch.device

    Returns
    -------
    dict with 'Wall' (n_units, n_chans, n_pcs), 'wPCA', 'wTEMP', 'whiten_mat'.

    """
    results_dir = Path(results_dir)
    for name in ['ops.npy', 'templates.npy']:
        if not (results_dir / name).exists():
            raise FileNotFoundError(
                f"'{name}' not found in previous_results_dir '{results_dir}'."
                )
    prior_ops = load_ops(results_dir / 'ops.npy', device=device)

    if not np.array_equal(np.asarray(prior_ops['probe']['chanMap']),
                          np.asarray(ops['probe']['chanMap'])):
        raise ValueError(
            'Previous results use a different channel map '
            f"({len(prior_ops['probe']['chanMap'])} channels) than this run "
            f"({len(ops['probe']['chanMap'])} channels). Use the same probe, "
            '`bad_channels` and `shank_idx` for both runs.'
            )
    for name, prev, cur in [
            ('fs', prior_ops['fs'], ops['fs']),
            ('nt', prior_ops['nt'], ops['nt']),
            ('n_pcs', prior_ops['settings']['n_pcs'], ops['settings']['n_pcs'])]:
        if prev != cur:
            raise ValueError(
                f'Previous results have {name}={prev}, but this run has {name}={cur}.'
                )

    wPCA = torch.as_tensor(prior_ops['wPCA']).float().to(device)
    wTEMP = torch.as_tensor(prior_ops['wTEMP']).float().to(device)
    whiten_mat = prior_ops['preprocessing']['whiten_mat'].float().to(device)

    if (results_dir / 'Wall.npy').exists():
        Wall = torch.from_numpy(np.load(results_dir / 'Wall.npy')).float()
    else:
        # Invert save_to_phy: templates[k,t,c] = sum_p Wall[k,c,p] * wPCA[p,t]
        templates = torch.from_numpy(np.load(results_dir / 'templates.npy')).float()
        Wall = torch.einsum('ktc, tp -> kcp', templates,
                            torch.linalg.pinv(wPCA.cpu()))

    return {'Wall': Wall.to(device), 'wPCA': wPCA, 'wTEMP': wTEMP,
            'whiten_mat': whiten_mat}
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_io.py -k prior -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add kilosort/io.py tests/test_io.py
git commit -m "feat: load and validate templates from a previous sorting result"
```

---

### Task 2: Dedupe candidates and re-estimate prior templates

**Files:**
- Modify: `kilosort/template_matching.py` (add after `postprocess_templates`, around line 153)
- Create: `tests/test_template_matching.py`

**Interfaces:**
- Produces:
  - `template_matching.drop_prior_duplicates(prior_Wall, cand_Wall, ops, r_thresh=0.9) -> torch.Tensor`
    - Returns the candidates whose max `CCG.similarity` to any prior template is `<= r_thresh`, on `cand_Wall.device`.
    - `ops` needs `'wPCA'` and `'nt'`.
  - `template_matching.reestimate_prior_templates(prior_Wall, clu, tF, iCC, iU) -> torch.Tensor`
    - Returns a CPU tensor of the same shape as `prior_Wall`.
    - Row `k` becomes `tF[clu == k].mean(0)` placed on channels `iCC[:, iU[k]]` (zeros elsewhere).
    - Rows with no spikes are copied unchanged.

- [ ] **Step 1: Write the failing tests** (`tests/test_template_matching.py`)

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_template_matching.py -v`
Expected: 3 FAIL with `AttributeError: ... has no attribute 'drop_prior_duplicates'` / `'reestimate_prior_templates'`

- [ ] **Step 3: Implement** (in `kilosort/template_matching.py`, after `postprocess_templates`)

```python
def drop_prior_duplicates(prior_Wall, cand_Wall, ops, r_thresh=0.9):
    """Remove candidate templates that duplicate a template from a previous sort."""
    N = prior_Wall.shape[0]
    if N == 0 or cand_Wall.shape[0] == 0:
        return cand_Wall
    both = torch.cat((prior_Wall.to(cand_Wall.device), cand_Wall), 0)
    sim = CCG.similarity(both, ops['wPCA'].contiguous(), nt=ops['nt'])
    keep = sim[N:, :N].max(1) <= r_thresh
    return cand_Wall[torch.from_numpy(keep).to(cand_Wall.device)]


def reestimate_prior_templates(prior_Wall, clu, tF, iCC, iU):
    """Average this session's features for each prior unit.

    Units with no spikes keep their old template so row k still means unit k.
    """
    W = prior_Wall.clone().cpu()
    iCC, iU = iCC.cpu(), iU.cpu()
    for k in range(W.shape[0]):
        ix = torch.from_numpy(clu == k)
        if ix.any():
            W[k] = 0
            W[k, iCC[:, iU[k]], :] = tF[ix].mean(0)
    return W
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_template_matching.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add kilosort/template_matching.py tests/test_template_matching.py
git commit -m "feat: helpers to dedupe candidates against and re-estimate prior templates"
```

---

### Task 3: Wire prior templates through the pipeline

**Files:**
- Modify: `kilosort/spikedetect.py:205-217` (start of `run`)
- Modify: `kilosort/run_kilosort.py`:
  - `run_kilosort` signature and docstring (lines ~33-200)
  - `_sort` (lines ~199-360)
  - `compute_preprocessing` (~585-660)
  - `detect_spikes` (~749-866)
  - new `cluster_spikes_with_prior` after `cluster_spikes` (~955)
- Create: `tests/test_prior_templates.py`

**Interfaces:**
- Consumes:
  - `io.load_prior_templates` (Task 1)
  - `template_matching.drop_prior_duplicates`, `template_matching.reestimate_prior_templates` (Task 2)
- Produces:
  - `run_kilosort(..., previous_results_dir=None)`
  - `compute_preprocessing(ops, device, tic0=np.nan, file_object=None, whiten_mat=None)`
  - `detect_spikes(..., prior_Wall=None)`
  - `cluster_spikes_with_prior(st, tF, ops, prior_Wall, device, bfile, tic0=np.nan, progress_bar=None, clear_cache=False, verbose=False) -> (clu, Wall, st, tF)`
  - `_prior_dir_for_shank(previous_results_dir, shank_idx) -> Path | None`
  - `ops['prior_basis'] = True` tells `spikedetect.run` to keep the existing `ops['wPCA']`/`ops['wTEMP']`.

- [ ] **Step 1: Write the failing tests** (`tests/test_prior_templates.py`)

```python
from pathlib import Path

import numpy as np
import torch

from kilosort import run_kilosort as rk


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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_prior_templates.py -v`
Expected: 3 FAIL with `AttributeError: module 'kilosort.run_kilosort' has no attribute 'cluster_spikes_with_prior'` / `'_prior_dir_for_shank'`

- [ ] **Step 3: Implement `cluster_spikes_with_prior` and `_prior_dir_for_shank`** (`kilosort/run_kilosort.py`, directly after `cluster_spikes`)

```python
def cluster_spikes_with_prior(st, tF, ops, prior_Wall, device, bfile,
                              tic0=np.nan, progress_bar=None,
                              clear_cache=False, verbose=False):
    """Keep prior unit IDs for spikes matched to prior templates and cluster
    the remaining spikes into new units numbered after the prior ones.

    Spikes with `st[:,1] < len(prior_Wall)` were matched to prior template
    `st[:,1]` by `template_matching.extract`, so that is their unit ID.

    """
    N = prior_Wall.shape[0]
    is_prior = st[:, 1] < N
    clu = np.full(len(st), -1, 'int32')
    clu[is_prior] = st[is_prior, 1].astype('int32')
    Wall_prior = template_matching.reestimate_prior_templates(
        prior_Wall, clu, tF, ops['iCC'], ops['iU']
        )
    logger.info(f'{int(is_prior.sum())} spikes assigned to '
                f'{np.unique(clu[is_prior]).size} of {N} previous units')

    Wall_new = torch.zeros((0, *prior_Wall.shape[1:]))
    new = ~is_prior
    if new.any():
        new_t = torch.from_numpy(new)
        try:
            clu_new, Wall_new, st_new, tF_new = cluster_spikes(
                st[new], tF[new_t], ops, device, bfile, tic0=tic0,
                progress_bar=progress_bar, clear_cache=clear_cache,
                verbose=verbose
                )
        except ValueError as e:
            logger.warning(f'Could not cluster {int(new.sum())} spikes that did '
                           f'not match previous units ({e}); dropping them.')
            keep = torch.from_numpy(is_prior)
            st, tF, clu = st[is_prior], tF[keep], clu[is_prior]
            Wall_new = torch.zeros((0, *prior_Wall.shape[1:]))
        else:
            st[new] = st_new
            tF[new_t] = tF_new
            clu[new] = clu_new + N

    Wall = torch.cat((Wall_prior.to(device), Wall_new.to(device)), 0)
    isort = np.argsort(st[:, 0], kind='stable')
    return clu[isort], Wall, st[isort], tF[torch.from_numpy(isort)]


def _prior_dir_for_shank(previous_results_dir, shank_idx):
    """Mirror `set_files`: results for shank i live in `results_dir/shank_i`."""
    if previous_results_dir is None:
        return None
    prior_dir = Path(previous_results_dir)
    if shank_idx is not None:
        prior_dir = prior_dir / f'shank_{shank_idx}'
    return prior_dir
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_prior_templates.py -v`
Expected: 3 PASS

- [ ] **Step 5: Keep the prior basis in `spikedetect.run`** (`kilosort/spikedetect.py`, replace the `if ops['settings']['templates_from_data']:` block at the top of `run`)

```python
    if ops.get('prior_basis', False):
        # wPCA/wTEMP were loaded from a previous result; prior templates
        # are expressed in this basis, so it must not be re-learned.
        logger.info('Using universal templates from previous results.')
    elif ops['settings']['templates_from_data']:
        logger.info('Re-computing universal templates from data.')
        # Determine templates and PC features from data.
        ops['wPCA'], ops['wTEMP'] = extract_wPCA_wTEMP(
            ops, bfile, nt=ops['nt'], twav_min=ops['nt0min'],
            Th_single_ch=ops['settings']['Th_single_ch'], nskip=25,
            device=device
            )
    else:
        logger.info('Using built-in universal templates.')
        # Use pre-computed templates.
        ops['wPCA'], ops['wTEMP'] = get_waves(ops, device=device)
```

- [ ] **Step 6: Optional whitening in `compute_preprocessing`** (`kilosort/run_kilosort.py`)

Change the signature to `def compute_preprocessing(ops, device, tic0=np.nan, file_object=None, whiten_mat=None):`. Add to the docstring Parameters: `whiten_mat : torch.Tensor; optional. If given, used instead of computing a whitening matrix from the data.` Then replace the `whiten_mat = preprocessing.get_whitening_matrix(...)` call with:

```python
    if whiten_mat is None:
        whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip,
                                                        nrange=whitening_range)
    else:
        logger.info('Using whitening matrix from previous results.')
```

- [ ] **Step 7: Prior templates in `detect_spikes`** (`kilosort/run_kilosort.py`)

Change the signature to `def detect_spikes(ops, device, bfile, tic0=np.nan, progress_bar=None, clear_cache=False, verbose=False, prior_Wall=None):`. Add to the docstring Parameters: `prior_Wall : torch.Tensor; optional. Templates from a previous sort. They are matched first (indices 0..N-1), together with any new candidate templates that don't duplicate them.` Replace the `st, tF, ops = template_matching.extract(ops, bfile, Wall3, ...)` call with:

```python
    U = Wall3
    if prior_Wall is not None:
        # LAYOUT: `Wall3` (and so `U`, as `template_matching.extract` expects) is
        # (units, n_pcs, Nchan) after `postprocess_templates`. `prior_Wall` and
        # both Task 2 helpers use the saved layout (units, Nchan, n_pcs).
        n_cand = Wall3.shape[0]
        cand = template_matching.drop_prior_duplicates(
            prior_Wall, Wall3.transpose(1, 2), ops
            )
        logger.info(f'{n_cand - cand.shape[0]} of {n_cand} new templates '
                    f'duplicate previous units; {cand.shape[0]} kept.')
        U = torch.cat((prior_Wall.transpose(1, 2).to(Wall3.device),
                       cand.transpose(1, 2)), 0).contiguous()
    st, tF, ops = template_matching.extract(
        ops, bfile, U, device=device, progress_bar=progress_bar
        )
```

- [ ] **Step 8: Thread `previous_results_dir` through `run_kilosort` and `_sort`** (`kilosort/run_kilosort.py`)

8a. Add `previous_results_dir=None` as the last parameter of `run_kilosort` (after `torch_thread_lim=None`). Document it after `torch_thread_lim` in the docstring:

```
    previous_results_dir : str or pathlib.Path; optional.
        Results folder of an earlier `run_kilosort` call on the same probe.
        Its templates are reused so that unit k here is unit k there; units
        not in the earlier result are numbered after them. The earlier
        whitening matrix and PC basis are reused. With `shank_idx`, the
        folder for shank i is `previous_results_dir/shank_i`.
```

8b. In the shank loop, pass it through:

```python
        ops, st, clu, tF, Wall, similar_templates, \
            is_ref, est_contam_rate, kept_spikes = _sort(
                _filename, _results_dir, _probe, settings, data_dtype, device,
                do_CAR, clear_cache, invert_sign, save_preprocessed_copy,
                verbose_log, save_extra_vars, file_object, progress_bar,
                previous_results_dir=_prior_dir_for_shank(previous_results_dir, idx),
            )
```

8c. Add `previous_results_dir=None` to the `_sort` signature after `gui_sorter=None`.

8d. In `_sort`, directly after the `logger.debug(f"Probe dictionary:...")` line, add:

```python
        prior = None
        if previous_results_dir is not None:
            prior = io.load_prior_templates(previous_results_dir, ops, device)
            ops['wPCA'], ops['wTEMP'] = prior['wPCA'], prior['wTEMP']
            ops['prior_basis'] = True
            logger.info(f"Reusing {prior['Wall'].shape[0]} templates from "
                        f"{previous_results_dir}")
```

8e. Change the `compute_preprocessing` call to:

```python
        ops = compute_preprocessing(
            ops, device, tic0=tic0, file_object=file_object,
            whiten_mat=None if prior is None else prior['whiten_mat']
            )
```

8f. Change the `detect_spikes` call to:

```python
        st,tF, Wall0, clu0 = detect_spikes(
            ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
            clear_cache=clear_cache, verbose=verbose_log,
            prior_Wall=None if prior is None else prior['Wall']
            )
```

8g. Replace the `cluster_spikes` call with:

```python
        if prior is None:
            clu, Wall, st, tF = cluster_spikes(
                st, tF, ops, device, bfile, tic0=tic0, progress_bar=progress_bar,
                clear_cache=clear_cache, verbose=verbose_log,
                )
        else:
            clu, Wall, st, tF = cluster_spikes_with_prior(
                st, tF, ops, prior['Wall'], device, bfile, tic0=tic0,
                progress_bar=progress_bar, clear_cache=clear_cache,
                verbose=verbose_log,
                )
```

- [ ] **Step 9: Run all fast tests and confirm nothing broke**

Run: `pytest tests -v`
Expected: all PASS. The slow pipeline tests are skipped without `--runslow`.

- [ ] **Step 10: Commit**

```bash
git add kilosort/run_kilosort.py kilosort/spikedetect.py tests/test_prior_templates.py
git commit -m "feat: run_kilosort reuses templates from previous_results_dir"
```

---

### Task 4: End-to-end check on real data (slow)

**Files:**
- Test: `tests/test_full_pipeline.py` (append)

**Interfaces:**
- Consumes:
  - `run_kilosort(..., results_dir=..., previous_results_dir=...)` (Task 3)
  - existing fixtures `data_directory`, `results_directory`, `torch_device`, `capture_mgr`, and pytest's `tmp_path`

- [ ] **Step 1: Find the recording length**

The test splits the recording in halves with `tmin`/`tmax` (seconds), which keeps each run short. Get the duration from the file size: `file_size / (385 channels * 2 bytes * 30000 Hz)`. The data downloads on first use (see `data_directory` in `tests/conftest.py`). Hard-code half of that duration, rounded down, as `HALF` in the test.

- [ ] **Step 2: Write the test** (append to `tests/test_full_pipeline.py`)

The test builds its own earlier result, so it does not depend on the downloaded reference results.

```python
@pytest.mark.slow
def test_pipeline_with_prior(data_directory, torch_device, tmp_path, capture_mgr):
    bin_file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    HALF = ...  # seconds: floor(file_size / (385 * 2 * 30000) / 2), see Step 1

    def sort(tmin, tmax, results_dir, prior=None):
        settings = {'n_chan_bin': 385, 'cluster_downsampling': 20,
                    'max_cluster_subset': None, 'tmin': tmin, 'tmax': tmax}
        out = run_kilosort(
            filename=bin_file, device=torch_device, settings=settings,
            probe_name='NeuroPix1_default.mat', results_dir=results_dir,
            previous_results_dir=prior,
            )
        _, _, clu, _, Wall, _, _, _, kept = out
        return clu[kept], Wall.shape[0]

    with capture_mgr.global_and_fixture_disabled():
        print('\nSession 1: first half, normal sort')
        clu1, n1 = sort(0, HALF, tmp_path / 'run1')
        print('Session 1b: same data, templates from session 1')
        clu1b, n1b = sort(0, HALF, tmp_path / 'run1b', prior=tmp_path / 'run1')
        print('Session 2: second half, templates from session 1b')
        clu2, n2 = sort(HALF, 2 * HALF, tmp_path / 'run2', prior=tmp_path / 'run1b')

    # Same data again: prior units keep their IDs, few new units appear
    assert n1b >= n1
    assert n1b - n1 <= 0.1 * n1
    assert np.mean(clu1b < n1) > 0.8
    # Large units keep roughly the same spike count
    counts1 = np.bincount(clu1, minlength=n1b)[:n1]
    counts1b = np.bincount(clu1b, minlength=n1b)[:n1]
    big = counts1 > 1000
    assert big.any()
    rel = np.abs(counts1b[big] - counts1[big]) / counts1[big]
    with capture_mgr.global_and_fixture_disabled():
        print(f'Relative spike-count change, {big.sum()} large units: '
              f'median {np.median(rel):.3f}, max {rel.max():.3f}')
    assert np.mean(rel < 0.2) >= 0.9

    # Different data, chained: still mostly prior unit IDs, no units removed
    assert n2 >= n1b
    assert np.mean(clu2 < n1b) > 0.7
```

- [ ] **Step 3: Run it**

Run: `pytest tests/test_full_pipeline.py --runslow -k prior -v -s` (add `--gpu` on a CUDA machine; on Apple Silicon the default device is fine).
Expected: PASS, with the printed median relative change well under 0.2. If an assertion fails, report the printed numbers rather than loosening the bounds, and say which of the three runs failed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_full_pipeline.py
git commit -m "test: end-to-end sort reusing previous templates, including chaining"
```

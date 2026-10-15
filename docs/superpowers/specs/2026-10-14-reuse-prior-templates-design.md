# Reuse prior templates across sessions — design

Date: 2026-10-14 · Branch: `acute`

## Goal

During acute experiments the probe barely moves between stimulus blocks/sessions.
We want a quick Kilosort run per session that keeps **cell identity** across runs:
unit `k` in session 2 is the same neuron as unit `k` in session 1.

`run_kilosort` gains an optional `previous_results_dir`. When given:

- Units from the prior sort keep their IDs (`0 … N-1`).
- Neurons not in the prior sort are still found, and numbered `N, N+1, …`.
- The output folder has the same format as a normal sort, so session 3 can point at
  session 2's output (chaining).

When `previous_results_dir=None`, behavior is unchanged.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| New neurons not in the prior sort | Match prior templates **and** find new units |
| Source of prior identity | Raw Kilosort output (`templates.npy`); Phy curation is ignored |
| Templates saved for matched units | Re-estimated from this session's spikes; units with no spikes keep their old template |
| Approach | Detect candidates normally, drop duplicates of prior templates, match prior + candidates in a single pass |

Assumptions: same probe, channel map, `bad_channels`/`shank_idx`, sampling rate, `nt`
and `n_pcs` between runs. Drift between sessions is small. Within-session drift
correction (`nblocks`) runs as configured.

## 1. Interface, loading, validation

- `run_kilosort(..., previous_results_dir=None)` is passed through to `_sort`.
- New function `io.load_prior_templates(previous_results_dir, device)` returns
  `Wall_prior` (units × channels × PCs), `wPCA`, `wTEMP`, `whiten_mat`, and the prior `ops`.
  - `ops.npy` is read with the existing `io.load_ops`. It supplies `wPCA`, `wTEMP`,
    `preprocessing['whiten_mat']`, the channel map, `fs`, `nt` and `n_pcs`.
  - `Wall_prior` is always rebuilt by projecting `templates.npy` onto `wPCA` (a
    pseudo-inverse). `Wall.npy` is deliberately ignored: `save_to_phy` never deletes an
    old one, so one left over in the folder from an earlier run can be stale.
    The round trip is exact (tested), so nothing is lost.
- Validation runs after `bad_channels` and `shank_idx` are applied. If the channel map,
  probe geometry (`xc`, `yc`), `fs`, `nt`, `n_pcs`, `invert_sign`, `do_CAR`,
  `highpass_cutoff` or `nt0min` differs from the current run, it raises `ValueError`
  naming the setting and showing both values.
- Shared basis and whitening:
  - `compute_preprocessing` still computes the high-pass filter etc., but its whitening
    matrix is replaced with the prior one.
  - `ops['wPCA']` and `ops['wTEMP']` are set to the prior values, and `spikedetect.run`
    must not re-learn them, even when `templates_from_data=True`.

## 2. Pipeline changes (`run_kilosort.py`)

When prior templates are given, `_sort` runs:

1. **Preprocessing and drift correction.** As today, with the prior whitening matrix.
2. **Candidates.** `detect_spikes(..., prior_Wall=...)` runs universal detection and the
   first clustering as normal, producing candidate templates `Wall3`.
3. **Dedupe.** `CCG.similarity` scores the candidates against the prior templates. Any
   candidate with similarity > 0.9 to any prior template is dropped. 0.9 is the cutoff
   `postprocess_templates` already uses.
4. **Single matching pass.** `U = cat(Wall_prior, surviving candidates)`, prior templates
   first. One call to `template_matching.extract`, so `st[:,1]` is each spike's row in `U`.
5. **Assign units.**
   - A spike with `st[:,1] < N` gets that unit ID directly.
   - The remaining spikes go through the existing `cluster_spikes` (final clustering and
     merging) on that subset only. The resulting IDs are offset by `N`.
   - If no spikes remain, this step is skipped.
6. **Re-estimate prior templates.** For prior unit `k` with spikes this session:
   `Wall[k, iCC[:, iU[k]], :] = tF[clu == k].mean(0)`. This is the same mean used in
   `clustering_qr.run`. Units with zero spikes keep their old template, so row `k` still
   means unit `k`.
7. **Save.** `Wall_final = cat(updated prior templates, new-unit templates)` goes to the
   unchanged `save_sorting`. The output is a normal results folder, and IDs `0…N-1` are
   the prior units.

Implementation style: `if prior is not None` branches inside `_sort` / `detect_spikes`,
plus one small helper for step 6. No separate parallel pipeline function.

## 3. Errors

- `ops.npy` or `templates.npy` missing → `FileNotFoundError` naming the file.
- Settings mismatch → `ValueError` (see §1).
- No spikes matched and no candidates → the existing "No spikes detected" error.
- Prior units with zero spikes this session: confirm during implementation that
  `save_to_phy` and `CCG.refract` cope with a template row that has no spikes. If not,
  add the minimal guard there.

## 4. Testing

Fast unit tests (no data download):

- Round trip `Wall` → `templates.npy` → `Wall` recovers the original
  (`tests/test_io.py`).
- Validation raises `ValueError` on a channel-count mismatch (`tests/test_io.py`).
- The step-6 re-estimation helper averages `tF` correctly and keeps old templates for
  units with no spikes (new file `tests/test_template_matching.py`).

Slow end-to-end test (`--runslow`, `tests/test_full_pipeline.py`), using the existing
downloaded dataset and its saved reference results as the prior sort:

- Re-sorting the same data with `previous_results_dir` assigns most spikes to prior IDs.
- Each large prior unit keeps its spike count within ~20%.
- Only a few new units appear.
- A second run that uses the first output as the prior also completes, which checks
  chaining.

## Out of scope

- Using Phy-curated clusters (`spike_clusters.npy`, `cluster_group.tsv`) as the prior.
- A matching-only fast mode (`find_new=False`). Add it if speed becomes the bottleneck.
- Correcting a rigid offset between sessions (probe moved between runs).
- GUI support (the GUI was removed on this branch).

import numpy as np
import pytest
import torch

from kilosort import run_kilosort


# Use `pytest --runslow` option to include this in tests.
@pytest.mark.slow
def test_pipeline(data_directory, results_directory, saved_ops, torch_device, capture_mgr):
    bin_file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    with pytest.raises(ValueError):
        # Should result in an error, since `n_chan_bin` isn't specified.
        ops, st, clu, _, _, _, _, _, kept_spikes = run_kilosort(
            settings={}, filename=bin_file, device=torch_device,
            probe_name='NeuroPix1_default.mat',
            )

    with capture_mgr.global_and_fixture_disabled():
        # NOTE: 'cluster_downsampling' and 'max_cluster_subset' are set to be
        #       equivalent to their default behavior prior to version 4.1.0,
        #       since that was how the test results were generated.
        print('\nStarting run_kilosort test...')
        ops, st, clu, _, _, _, _, _, kept_spikes = run_kilosort(
            filename=bin_file, device=torch_device,
            settings={'n_chan_bin': 385, 'cluster_downsampling': 20,
                      'max_cluster_subset': None},
            probe_name='NeuroPix1_default.mat',
            verbose_console=True
            )

    st = st[kept_spikes,0]  # only first column is spike times
    clu = clu[kept_spikes]
        
    # Check that spike times and spike cluster assignments match
    st_load = np.load(results_directory / 'spike_times.npy')
    clu_load = np.load(results_directory / 'spike_clusters.npy')
    saved_yblk = saved_ops['yblk']
    saved_dshift = saved_ops['dshift']
    saved_iKxx = saved_ops['iKxx']

    # Datashift output
    # assert np.allclose(saved_yblk, ops['yblk'])
    # TODO: Why is this resulting in small deviations on different systems?
    # assert np.allclose(saved_dshift, ops['dshift'])
    # TODO: Why is this suddenly getting a dimension mismatch?
    # assert torch.allclose(saved_iKxx, ops['iKxx'])

    # Final spike/neuron readout
    # Less than 2.5% difference in spike count, 5% difference in number of units
    # TODO: Make sure these are reasonable error bounds
    spikes_error = np.abs(st.size - st_load.size)/np.max([st.size, st_load.size])
    with capture_mgr.global_and_fixture_disabled():
        print(f'Proportion difference in total spike count: {spikes_error}')
        print(f'Count from run_kilosort: {st.size}')
        print(f'Count from saved test results: {st_load.size}')

    n = np.unique(clu).size
    n_load = np.unique(clu_load).size
    unit_count_error = np.abs(n - n_load)/np.max([n, n_load])
    with capture_mgr.global_and_fixture_disabled():
        print(f'Proportion difference in number of units: {unit_count_error}')
        print(f'Number of units from run_kilosort: {n}')
        print(f'Number of units from saved test results: {n_load}')

    assert spikes_error <= 0.025
    assert unit_count_error <= 0.05


@pytest.mark.slow
def test_pipeline_with_prior(data_directory, torch_device, tmp_path, capture_mgr):
    bin_file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    HALF = 22  # seconds: floor(1039500000 / (385 * 2 * 30000) / 2), file is 45 s

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
    # First-clustering candidates merged away in session 1's final merge can
    # resurface as new units (measured 11/100 with HALF=22).
    assert n1b - n1 <= 0.15 * n1
    assert np.mean(clu1b < n1) > 0.8
    # Large units keep roughly the same spike count
    counts1 = np.bincount(clu1, minlength=n1b)[:n1]
    counts1b = np.bincount(clu1b, minlength=n1b)[:n1]
    # 22 s of data: only 1 unit has >1000 spikes, 74 have >200
    big = counts1 > 200
    assert big.any()
    rel = np.abs(counts1b[big] - counts1[big]) / counts1[big]
    with capture_mgr.global_and_fixture_disabled():
        print(f'Relative spike-count change, {big.sum()} large units: '
              f'median {np.median(rel):.3f}, max {rel.max():.3f}')
    assert np.mean(rel < 0.2) >= 0.9

    # Different data, chained: still mostly prior unit IDs, no units removed
    assert n2 >= n1b
    assert np.mean(clu2 < n1b) > 0.7

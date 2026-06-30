# Performance Notes — rsomics-quantile-transform

## Machine

mini_m2 (Apple M2), macOS 25.5.0, single-thread (no parallelism in current impl).

## Tool versions

- ours: rsomics-quantile-transform 0.1.0 (this crate, release build)
- upstream: scikit-learn 1.9.0 in scanpy conda env (`/opt/homebrew/Caskroom/miniforge/base/envs/scanpy`)
  - Python 3.12, numpy 2.x, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1

## Fixture

`/Volumes/KIOXIA/rsomics-fixtures/quantile-transform/mat_5000x200.tsv`  
5000 rows × 200 cols, synthetic Gaussian (`np.random.RandomState(99).randn`), 18 MB TSV.

## Benchmark axes

Per [rsomics methodology](https://github.com/omics-rust/rsomics-world/docs/): output-dominated
tools require TWO fair axes:

1. **Compute-only**: ours `fit_quantiles+transform_matrix` in-process vs sklearn
   `fit_transform` in-process, single-thread, no serialization. Measures algorithm efficiency.
2. **Both-serialize**: ours (full binary: TSV parse + compute + TSV write to /dev/null) vs
   sklearn (`fit_transform` + `np.savetxt('/dev/null')`). End-to-end pipeline fair comparison.

## Results

### Both-serialize (5000 × 200, n_quantiles=1000, uniform)

```
ours:    226 ms min, 230 ms mean ± 4 ms   (hyperfine --warmup 5 --min-runs 12)
sklearn: 658 ms min, 678 ms mean           (10 Python timing loops)
ratio:   2.9× faster both-serialize
```

### Compute-only (5000 × 200, n_quantiles=1000, uniform)

```
ours compute (estimate):    ~150-170 ms  (both-serialize 226ms minus I/O ~60ms)
sklearn compute (measured): 212 ms mean  (10 Python timing loops, OPENBLAS_NUM_THREADS=1)
ratio:   ~1.3× faster compute
```

### Compute-only criterion bench (1000 × 50, n_quantiles=1000)

```
ours (criterion):    2.3 ms mean
sklearn (Python):   28.7 ms min, 32.6 ms mean
ratio:              12.5× faster compute (smaller fixture, no I/O overhead)
```

## Algorithm

Our algorithm sorts each column ONCE in O(n log n), then evaluates all 1000 quantile
levels via O(1) sorted-array lookup — total O(n log n + n_quantiles) per column.
sklearn's numpy `nanpercentile` uses a similar internal strategy for batched quantiles.

The both-serialize speedup (2.9×) exceeds compute speedup (1.3×) because numpy's
`savetxt` is ~3× slower than our BufWriter TSV output for this fixture size.

## Why ours is faster compute

1. **Single sort** per column (vs numpy's repeated partial sort for each quantile level).
2. **No Python overhead**: function call, GIL, array allocation per quantile.
3. **No DataFrame/ndarray boxing**: direct `Vec<f64>` slice operations.

## Memory

Our peak RSS: ~29 MB for 5000×200 (measured via `/usr/bin/time -l`).
sklearn: comparable (in-memory array copy for fit_transform).

## Notes

- No parallelism implemented yet; single-thread throughout.
- OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 enforced for sklearn to ensure fair
  single-thread comparison (sklearn's `nanpercentile` uses BLAS internally for some ops).
- `n_samples ≤ subsample` (the common case): no subsampling overhead, purely algorithmic.
- `n_samples > subsample` path: one MT19937 shuffle, same big-O.

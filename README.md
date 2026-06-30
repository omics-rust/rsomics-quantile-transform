# rsomics-quantile-transform

Quantile transformer CLI — independent Rust reimplementation of scikit-learn's
`QuantileTransformer`. Maps each column of a feature matrix to a uniform or normal
distribution by its empirical quantiles.

```
rsomics-quantile-transform [OPTIONS] [MATRIX]
```

## Usage

```
# Uniform output (default): map features to [0, 1]
rsomics-quantile-transform matrix.tsv --n-quantiles 1000

# Normal output: map features to a standard normal distribution
rsomics-quantile-transform matrix.tsv --output-distribution normal

# JSON envelope
rsomics-quantile-transform matrix.tsv --json

# Read from stdin
cat matrix.tsv | rsomics-quantile-transform -
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--n-quantiles N` | 1000 | Quantile landmarks (clamped to n_samples) |
| `--output-distribution` | `uniform` | `uniform` or `normal` |
| `--subsample N` | 10000 | Max rows for quantile estimation; 0 = no limit |
| `--random-state SEED` | 0 | RNG seed for subsampling |
| `-t / --threads` | 1 | (CommonFlags) |
| `--json` | off | JSON envelope output |

### Input format

Tab-separated `n×d` matrix. A leading tab (empty top-left cell) marks a header row
and enables row names in the first column. `NA`/`NaN`/empty cells → NaN (passed through).
Headerless matrices are accepted; rows/columns numbered from 1.

## Accuracy

- **Uniform output**: bit-exact at low / clamped `n_quantiles`; at high `n_quantiles` a small
  fraction of elements (≈0.03% measured) drift by ≤1 ULP, where the `0.5*(interp(x) - interp(-x))`
  averaging and numpy's FMA `np.interp` round their last bit differently. Type-7 `np.nanpercentile`,
  `np.linspace`, and the interp FMA order are otherwise replicated at the floating-point level.
- **Normal output**: ≤ 1e-12 relative vs scikit-learn. Cross-architecture (arm vs x86)
  Cephes `ndtri` transcendental floors at ~1 ULP difference, bounded by 1e-12 relative.
- **Subsampling** (`n_samples > --subsample`): BIT-EXACT. The MT19937 shuffle matches
  `np.random.RandomState(seed).shuffle(np.arange(n))` bit-for-bit, and the subsequent
  `np.interp` FMA evaluation order is replicated exactly.

## Performance (mini_m2, single-thread, 5000 × 200 matrix, n_quantiles=1000)

| Axis | Ours | sklearn 1.9.0 | Ratio |
|---|---|---|---|
| Both-serialize | 226 ms | 658 ms | **2.9×** |
| Compute-only (estimate) | ~155 ms | 212 ms | **~1.4×** |

Both-serialize: ours (TSV read + compute + TSV write) vs sklearn (`fit_transform` + `np.savetxt`).
See `PERF_NOTES.md` for provenance.

## Origin

This crate is an independent Rust reimplementation of `QuantileTransformer` from scikit-learn
based on:

- The scikit-learn 1.9.0 BSD-3-Clause source (`sklearn/preprocessing/_data.py`) — reading
  and citing the MIT/BSD source is required and expected per rsomics methodology.
- The numpy type-7 percentile and `np.interp` linear interpolation specifications.
- Black-box behavior verified via golden fixtures generated from real sklearn 1.9.0 output.

Key implementation choices informed by source reading:
- `references_ = np.linspace(0, 1, n_quantiles_)` — `i * step` not `i/n-1` to match numpy bits.
- `np.nanpercentile` virtual index uses `(n-1) * (q/100)` not `(n-1)*q/100` (parentheses matter).
- `np.interp` uses `slope.mul_add(x - xp[i], fp[i])` (FMA with `slope=(fp[j+1]-fp[j])/(xp[j+1]-xp[j])`).
- `resample(X, replace=False)` shuffles ALL rows once then takes `[:k]` — shared across all columns.

No GPL source was used. License: MIT OR Apache-2.0.  
Upstream credit: [scikit-learn](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause).

## Cephes ndtri

The `ndtri` (inverse normal CDF) is ported from Cephes (S. L. Moshier), the same
implementation used by `scipy.special.ndtri` / `scipy.stats.norm.ppf`. Coefficients
are transcribed verbatim from Cephes source at full source precision.

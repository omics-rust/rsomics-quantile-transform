//! Type-7 `np.nanpercentile` and quantile fitting — sklearn `QuantileTransformer._dense_fit`.
//!
//! `np.nanpercentile` (linear = type-7) places the virtual index `h = (n−1)·q/100` over
//! the sorted non-NaN survivors and linearly interpolates the bracketing order statistics.
//! sklearn builds `references_ = linspace(0, 1, n_quantiles_)`, calls
//! `np.nanpercentile(X, references * 100, axis=0)`, then `np.maximum.accumulate`
//! along axis=0 to force monotonicity (guards against floating-point reversals at
//! repeated values).
//!
//! When n_samples > subsample, sklearn resamples the **entire matrix** (shared row
//! indices for all columns) via one `resample(X, replace=False)` call. We replicate
//! that by drawing indices once then extracting per-column subsets.

use crate::rng::{Mt19937, shuffle_arange};

/// Type-7 quantile of `xs` at probability `p ∈ [0,1]`. `xs` is reordered in place.
#[cfg(test)]
fn quantile_type7(xs: &mut [f64], p: f64) -> f64 {
    let n = xs.len();
    if n == 1 {
        return xs[0];
    }
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 >= n {
        return *xs.select_nth_unstable_by(n - 1, f64::total_cmp).1;
    }
    let (_, &mut kth, right) = xs.select_nth_unstable_by(lo, f64::total_cmp);
    let next = right.iter().copied().fold(f64::INFINITY, f64::min);
    kth + frac * (next - kth)
}

/// `np.nanpercentile(col, q)` with `q` in percent. All-NaN → NaN.
/// Used for single-quantile queries (unit tests / internal use).
#[cfg(test)]
fn nanpercentile(col: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = col.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    quantile_type7(&mut v, q / 100.0)
}

/// `np.nanpercentile(col, references * 100)` for all references in one pass.
///
/// Sorts the finite values ONCE then evaluates every quantile via the type-7 formula
/// on the sorted array in O(n_quantiles) lookups — vs O(n × n_quantiles) if we called
/// quickselect for each quantile individually. This is the critical performance path.
///
/// The virtual index `h` is computed as `(n-1) * q / 100` where `q = r * 100`,
/// matching numpy's internal FP evaluation order so quantile values are bit-identical.
fn nanpercentile_all(col: &[f64], references: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = col.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return vec![f64::NAN; references.len()];
    }
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    let nm1 = (n - 1) as f64;
    references
        .iter()
        .map(|&r| {
            // sklearn calls np.nanpercentile(col, references*100), so q = r*100.
            // numpy computes h = (n-1) * (q/100), with parentheses — NOT (n-1)*q/100.
            // The round-trip r→r*100→÷100 can differ by up to 1 ULP, which changes lo.
            let q = r * 100.0;
            let h = nm1 * (q / 100.0);
            let lo = h.floor() as usize;
            let frac = h - lo as f64;
            if lo + 1 >= n {
                return v[n - 1];
            }
            v[lo] + frac * (v[lo + 1] - v[lo])
        })
        .collect()
}

/// Fit `quantiles_` for one (already-subsampled) column.
/// Returns a `Vec` of length `references.len()` after `np.maximum.accumulate`.
fn fit_column(col: &[f64], references: &[f64]) -> Vec<f64> {
    let mut q = nanpercentile_all(col, references);

    // np.maximum.accumulate along quantile axis — enforces monotonicity.
    let mut running_max = f64::NEG_INFINITY;
    for v in &mut q {
        if !v.is_nan() {
            if *v < running_max {
                *v = running_max;
            } else {
                running_max = *v;
            }
        }
    }

    q
}

/// Fit quantile tables for every column. Returns `(references, quantiles)` where
/// `quantiles[j]` is the `n_quantiles`-length vector for column `j`.
///
/// `n_quantiles` is clamped to `n_samples` per sklearn's `fit()` logic.
pub fn fit_quantiles(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    n_quantiles_req: usize,
    subsample: Option<usize>,
    seed: u64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n_quantiles = n_quantiles_req.min(n_rows).max(1);
    // np.linspace(0, 1, n_quantiles): multiply i by step (not divide i by n-1)
    // to match numpy's bit pattern (7.0/9.0 != 7.0*(1.0/9.0) at i=7, n=10).
    let step = if n_quantiles > 1 {
        1.0 / (n_quantiles - 1) as f64
    } else {
        0.0
    };
    let references: Vec<f64> = (0..n_quantiles)
        .map(|i| {
            if i == n_quantiles - 1 {
                1.0
            } else {
                i as f64 * step
            }
        })
        .collect();

    // Pre-compute shared subsample row indices when n_rows > k (one shuffle, all columns).
    let shared_indices: Option<Vec<usize>> = subsample.and_then(|k| {
        if n_rows > k {
            let mut rng = Mt19937::seed(seed as u32);
            let all = shuffle_arange(&mut rng, n_rows);
            Some(all[..k].to_vec())
        } else {
            None
        }
    });

    let quantiles: Vec<Vec<f64>> = (0..n_cols)
        .map(|j| {
            let col: Vec<f64> = match &shared_indices {
                Some(idxs) => idxs.iter().map(|&i| data[i * n_cols + j]).collect(),
                None => (0..n_rows).map(|i| data[i * n_cols + j]).collect(),
            };
            fit_column(&col, &references)
        })
        .collect();

    (references, quantiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn nanpercentile_type7() {
        let c = [1.0, 2.0, 3.0, 4.0, 5.0];
        close(nanpercentile(&c, 25.0), 2.0);
        close(nanpercentile(&c, 50.0), 3.0);
        close(nanpercentile(&c, 75.0), 4.0);
        close(nanpercentile(&c, 10.0), 1.4);
        close(nanpercentile(&c, 0.0), 1.0);
        close(nanpercentile(&c, 100.0), 5.0);
    }

    #[test]
    fn nanpercentile_unsorted() {
        close(nanpercentile(&[7.0, 3.0, 9.0, 1.0, 5.0], 10.0), 1.8);
        close(nanpercentile(&[7.0, 3.0, 9.0, 1.0, 5.0], 75.0), 7.0);
    }

    #[test]
    fn fit_column_basic() {
        // 5 samples, 5 quantiles → references=[0,0.25,0.5,0.75,1] (step=1/4 exact)
        // np.nanpercentile([1,2,3,4,5], [0,25,50,75,100]) = [1,2,3,4,5]
        let refs: Vec<f64> = (0..5).map(|i| i as f64 / 4.0).collect();
        let q = fit_column(&[1.0, 2.0, 3.0, 4.0, 5.0], &refs);
        for (got, want) in q.iter().zip(&[1.0, 2.0, 3.0, 4.0, 5.0]) {
            close(*got, *want);
        }
    }

    #[test]
    fn monotonic_accumulate() {
        let refs = vec![0.0, 0.5, 1.0];
        let q = fit_column(&[5.0, 5.0, 5.0], &refs);
        assert!(q.windows(2).all(|w| w[0] <= w[1]));
    }
}

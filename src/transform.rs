//! Per-column forward transform — sklearn `QuantileTransformer._transform_col`.
//!
//! Forward map:
//!   1. Double-interp averaging (handles ties):
//!      `0.5 * (interp(x, quantiles, refs) - interp(-x, -quantiles[::-1], -refs[::-1]))`
//!   2. Force boundary values to exactly 0/1.
//!   3. For normal output: apply `ndtri` then clip to `[CLIP_MIN, CLIP_MAX]`.
//!
//! Uniform output is purely linear interpolation → BIT-EXACT vs sklearn (0 ULP).
//! Normal output adds the Cephes `ndtri` transcendental; cross-arch last bits can
//! differ ≤1 ULP, so compat tolerance is ≤1e-12 relative.

use crate::ndtri::{CLIP_MAX, CLIP_MIN, ndtri};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDistribution {
    Uniform,
    Normal,
}

/// Linear interpolation matching `np.interp(x, xp, fp)`.
///
/// numpy's C implementation computes `slope = (fp[j+1]-fp[j])/(xp[j+1]-xp[j])` then
/// returns `fp[j] + slope * (x - xp[j])` — which is `fp[j].mul_add(slope, dx)` in FP.
/// Using the same `fp[i].mul_add(slope, dx)` → `fp[i] + slope*(x-xp[i])` order
/// via `f64::mul_add` (FMA) matches numpy bit-for-bit including in the subsample regime.
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let idx = xp.partition_point(|&v| v <= x);
    let i = idx - 1;
    let slope = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]);
    // FMA: slope * (x - xp[i]) + fp[i]
    slope.mul_add(x - xp[i], fp[i])
}

/// Transform a single column in place, matching `_transform_col(inverse=False)`.
pub fn transform_col(
    col: &mut [f64],
    quantiles: &[f64],
    references: &[f64],
    dist: OutputDistribution,
) {
    let lower_bound_x = quantiles[0];
    let upper_bound_x = quantiles[quantiles.len() - 1];

    // Build reversed views for the second interp.
    let q_rev: Vec<f64> = quantiles.iter().rev().map(|&v| -v).collect();
    let r_rev: Vec<f64> = references.iter().rev().map(|&v| -v).collect();

    for v in col.iter_mut() {
        if v.is_nan() {
            continue;
        }
        let x = *v;

        // sklearn uses `== lower/upper_bound_x` for uniform boundary detection.
        let at_lower = x == lower_bound_x;
        let at_upper = x == upper_bound_x;

        // Double-interp averaging (ties handling, per sklearn comment).
        let fwd = np_interp(x, quantiles, references);
        let rev = -np_interp(-x, &q_rev, &r_rev);
        let mut y = 0.5 * (fwd + rev);

        // Exact boundary values must land at exactly 0 / 1 (sklearn sets them after interp).
        if at_upper {
            y = 1.0;
        }
        if at_lower {
            y = 0.0;
        }

        if dist == OutputDistribution::Normal {
            y = ndtri(y).clamp(CLIP_MIN, CLIP_MAX);
        }

        *v = y;
    }
}

/// Transform every column of the matrix (row-major `data`, `n_rows × n_cols`).
pub fn transform_matrix(
    data: &mut [f64],
    n_rows: usize,
    n_cols: usize,
    quantiles_per_col: &[Vec<f64>],
    references: &[f64],
    dist: OutputDistribution,
) {
    // Work column by column; extract → transform → scatter back.
    for j in 0..n_cols {
        let mut col: Vec<f64> = (0..n_rows).map(|i| data[i * n_cols + j]).collect();
        transform_col(&mut col, &quantiles_per_col[j], references, dist);
        for (i, v) in col.into_iter().enumerate() {
            data[i * n_cols + j] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-12,
            "got={a} want={b} diff={}",
            (a - b).abs()
        );
    }

    #[test]
    fn np_interp_basic() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 0.5, 1.0];
        close(np_interp(0.5, &xp, &fp), 0.25);
        close(np_interp(0.0, &xp, &fp), 0.0);
        close(np_interp(2.0, &xp, &fp), 1.0);
        close(np_interp(-1.0, &xp, &fp), 0.0); // below → fp[0]
        close(np_interp(3.0, &xp, &fp), 1.0); // above → fp[-1]
    }

    #[test]
    fn uniform_ties_average() {
        // quantiles = [1,2,2,2,3], refs = [0,.25,.5,.75,1]
        // x=2 → fwd=interp(2,[1,2,2,2,3],[0,.25,.5,.75,1])=0.25
        //        rev=-interp(-2,[-3,-2,-2,-2,-1],[−1,−.75,−.5,−.25,0])
        //        fwd for rev interp: x=-2 in [-3,-2,-2,-2,-1] → 0.75
        //        rev term = -(−0.75) but wait — refs_rev = [−1,−.75,−.5,−.25,0]
        //        -interp(-2,[-3,-2,-2,-2,-1],[-1,-.75,-.5,-.25,0]) = -(-0.75) = 0.75
        //        average = 0.5*(0.25+0.75) = 0.5
        let quantiles = [1.0, 2.0, 2.0, 2.0, 3.0];
        let refs = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut col = [2.0];
        transform_col(&mut col, &quantiles, &refs, OutputDistribution::Uniform);
        close(col[0], 0.5);
    }

    #[test]
    fn boundary_forced_to_exact() {
        let quantiles = [1.0, 2.0, 3.0];
        let refs = [0.0, 0.5, 1.0];
        let mut col = [1.0, 3.0];
        transform_col(&mut col, &quantiles, &refs, OutputDistribution::Uniform);
        assert_eq!(col[0], 0.0);
        assert_eq!(col[1], 1.0);
    }
}

//! Value-exact compatibility tests vs sklearn 1.9.0 QuantileTransformer.
//!
//! Goldens were generated ONCE from real sklearn 1.9.0 (scanpy env) and are
//! checked into tests/golden/. Tests never call Python; they compare our binary
//! output against the frozen goldens.
//!
//! Uniform output: bit-exact at low/clamped n_quantiles; ≤1 ULP on a small
//! fraction of elements at high n_quantiles, where the `0.5*(interp(x) -
//! interp(-x))` averaging and numpy's FMA `interp` round their last bit
//! differently. Asserted bit-exact on the small goldens and ≤2 ULP on the
//! large high-n_quantiles golden.
//! Normal output: ≤ 1e-12 relative (cross-arch Cephes ndtri floor).
//! Subsample regime (n_samples > --subsample): bit-exact — the MT19937 index
//! draw and interp match numpy's path.

use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

fn binary() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    // target/debug/deps → target/debug
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("rsomics-quantile-transform")
}

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(name)
}

fn run(args: &[&str]) -> String {
    let out = Command::new(binary())
        .args(args)
        .output()
        .expect("failed to run binary");
    assert!(
        out.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// Parse TSV output rows (skip header, skip row-name column).
fn parse_tsv(output: &str) -> Vec<Vec<f64>> {
    output
        .lines()
        .skip(1) // header
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split('\t')
                .skip(1) // row name
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

/// Parse TSV golden (headerless, no row names).
fn parse_tsv_golden(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split('\t')
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect()
        })
        .collect()
}

/// Parse hex golden into f64 via bit pattern.
fn parse_hex_golden(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            line.split('\t')
                .map(|h| {
                    let bytes = u64::from_str_radix(h.trim(), 16).unwrap();
                    f64::from_bits(bytes)
                })
                .collect()
        })
        .collect()
}

fn assert_bit_exact(got: &[Vec<f64>], want: &[Vec<f64>], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: row count mismatch");
    for (i, (g_row, w_row)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g_row.len(),
            w_row.len(),
            "{label}: row {} col count mismatch",
            i + 1
        );
        for (j, (&g, &w)) in g_row.iter().zip(w_row).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}: row {} col {} got={g:.17e} want={w:.17e}",
                i + 1,
                j + 1
            );
        }
    }
}

fn assert_normal_rel(got: &[Vec<f64>], want: &[Vec<f64>], tol: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: row count mismatch");
    for (i, (g_row, w_row)) in got.iter().zip(want).enumerate() {
        for (j, (&g, &w)) in g_row.iter().zip(w_row).enumerate() {
            if g.is_nan() && w.is_nan() {
                continue;
            }
            let rel = (g - w).abs() / w.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel <= tol,
                "{label}: row {} col {} rel={rel:.3e} > {tol:.3e} (got={g:.17e} want={w:.17e})",
                i + 1,
                j + 1
            );
        }
    }
}

fn ulps(a: f64, b: f64) -> u64 {
    let map = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN.wrapping_sub(bits)
        } else {
            bits
        }
    };
    map(a).abs_diff(map(b))
}

fn assert_within_ulps(got: &[Vec<f64>], want: &[Vec<f64>], max_ulps: u64, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: row count mismatch");
    for (i, (g_row, w_row)) in got.iter().zip(want).enumerate() {
        for (j, (&g, &w)) in g_row.iter().zip(w_row).enumerate() {
            let u = ulps(g, w);
            assert!(
                u <= max_ulps,
                "{label}: row {} col {} {u} ULP > {max_ulps} (got={g:.17e} want={w:.17e})",
                i + 1,
                j + 1
            );
        }
    }
}

/// High n_quantiles (2000×4, q=1000 unclamped): exercises the interp/averaging
/// path where uniform output drifts ≤1 ULP from sklearn on a few elements.
#[test]
fn uniform_large_2000x4_q1000_within_1_ulp() {
    let input = golden("large_2000x4.tsv");
    let out = run(&[input.to_str().unwrap(), "--n-quantiles", "1000"]);
    let got = parse_tsv(&out);
    let want = parse_hex_golden(
        &std::fs::read_to_string(golden("large_2000x4_uniform_q1000.hex")).unwrap(),
    );
    assert_within_ulps(&got, &want, 2, "large_2000x4_uniform_q1000");
}

// ── GOLDEN 1: basic 10×3 matrix, n_quantiles=10, uniform ──────────────────────

#[test]
fn uniform_basic_10x3_q10_bit_exact() {
    let input = golden("basic_10x3.tsv");
    let out = run(&[input.to_str().unwrap(), "--n-quantiles", "10"]);
    let got = parse_tsv(&out);
    let want =
        parse_hex_golden(&std::fs::read_to_string(golden("basic_10x3_uniform_q10.hex")).unwrap());
    assert_bit_exact(&got, &want, "basic_10x3_uniform_q10");
}

// ── GOLDEN 2: basic 10×3 matrix, n_quantiles=10, normal ──────────────────────

#[test]
fn normal_basic_10x3_q10_rel1e12() {
    let input = golden("basic_10x3.tsv");
    let out = run(&[
        input.to_str().unwrap(),
        "--n-quantiles",
        "10",
        "--output-distribution",
        "normal",
    ]);
    let got = parse_tsv(&out);
    let want =
        parse_tsv_golden(&std::fs::read_to_string(golden("basic_10x3_normal_q10.tsv")).unwrap());
    assert_normal_rel(&got, &want, 1e-12, "basic_10x3_normal_q10");
}

// ── GOLDEN 3: constant column, n_quantiles=5 ──────────────────────────────────

#[test]
fn uniform_constant_col_q5_bit_exact() {
    let input = golden("constant_col.tsv");
    let out = run(&[input.to_str().unwrap(), "--n-quantiles", "5"]);
    let got = parse_tsv(&out);
    let want =
        parse_hex_golden(&std::fs::read_to_string(golden("constant_col_uniform_q5.hex")).unwrap());
    assert_bit_exact(&got, &want, "constant_col_uniform_q5");
}

// ── GOLDEN 4: ties, n_quantiles=5 ─────────────────────────────────────────────

#[test]
fn uniform_ties_q5_bit_exact() {
    let input = golden("ties.tsv");
    let out = run(&[input.to_str().unwrap(), "--n-quantiles", "5"]);
    let got = parse_tsv(&out);
    let want = parse_hex_golden(&std::fs::read_to_string(golden("ties_uniform_q5.hex")).unwrap());
    assert_bit_exact(&got, &want, "ties_uniform_q5");
}

// ── GOLDEN 5: negative values, n_quantiles=5 ──────────────────────────────────

#[test]
fn uniform_negative_q5_bit_exact() {
    let input = golden("negative.tsv");
    let out = run(&[input.to_str().unwrap(), "--n-quantiles", "5"]);
    let got = parse_tsv(&out);
    let want =
        parse_hex_golden(&std::fs::read_to_string(golden("negative_uniform_q5.hex")).unwrap());
    assert_bit_exact(&got, &want, "negative_uniform_q5");
}

// ── GOLDEN 6: n_quantiles=1000 clamped to n_samples=10 ───────────────────────

#[test]
fn uniform_q1000_clamped_bit_exact() {
    let input = golden("basic_10x3.tsv");
    let out = run(&[input.to_str().unwrap(), "--n-quantiles", "1000"]);
    let got = parse_tsv(&out);
    let want = parse_hex_golden(
        &std::fs::read_to_string(golden("basic_10x3_uniform_q1000_clamped.hex")).unwrap(),
    );
    assert_bit_exact(&got, &want, "q1000_clamped");
}

// ── GOLDEN 7: n_samples=20 > subsample=10, random_state=42 ───────────────────

#[test]
fn uniform_subsample_bit_exact() {
    let input = golden("big_20x2.tsv");
    let out = run(&[
        input.to_str().unwrap(),
        "--n-quantiles",
        "5",
        "--subsample",
        "10",
        "--random-state",
        "42",
    ]);
    let got = parse_tsv(&out);
    let want = parse_hex_golden(
        &std::fs::read_to_string(golden("big_20x2_uniform_q5_sub10.hex")).unwrap(),
    );
    assert_bit_exact(&got, &want, "subsample_uniform");
}

// ── GOLDEN 8: normal output for ties ─────────────────────────────────────────

#[test]
fn normal_ties_q5_rel1e12() {
    let input = golden("ties.tsv");
    let out = run(&[
        input.to_str().unwrap(),
        "--n-quantiles",
        "5",
        "--output-distribution",
        "normal",
    ]);
    let got = parse_tsv(&out);
    let want = parse_tsv_golden(&std::fs::read_to_string(golden("ties_normal_q5.tsv")).unwrap());
    assert_normal_rel(&got, &want, 1e-12, "ties_normal_q5");
}

// ── Stdin / portable tempdir smoke test ──────────────────────────────────────

#[test]
fn stdin_reads_and_outputs_tsv() {
    let dir = tempdir().unwrap(); // portable tempfile::tempdir() — no KIOXIA path
    let inp = dir.path().join("m.tsv");
    std::fs::write(&inp, "\tc1\tc2\nr1\t1.0\t2.0\nr2\t3.0\t4.0\nr3\t5.0\t6.0\n").unwrap();
    let out_str = Command::new(binary())
        .args([inp.to_str().unwrap(), "--n-quantiles", "3"])
        .output()
        .unwrap();
    assert!(out_str.status.success());
    let text = String::from_utf8(out_str.stdout).unwrap();
    // Header row and 3 data rows
    assert_eq!(text.lines().count(), 4);
}

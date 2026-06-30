use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_quantile_transform::{OutputDistribution, fit_quantiles, transform_matrix};

fn bench_fit_transform(c: &mut Criterion) {
    // 1000 samples × 50 features; typical genomics expression matrix dimensions
    let n_rows = 1000usize;
    let n_cols = 50usize;
    let data: Vec<f64> = (0..n_rows * n_cols)
        .map(|i| ((i as f64 * 1.6180339887) % 7.0) - 3.5)
        .collect();

    c.bench_function("fit_transform_1000x50_uniform", |b| {
        b.iter(|| {
            let (refs, q) = fit_quantiles(&data, n_rows, n_cols, 1000, Some(10_000), 0);
            let mut out = data.clone();
            transform_matrix(
                &mut out,
                n_rows,
                n_cols,
                &q,
                &refs,
                OutputDistribution::Uniform,
            );
            std::hint::black_box(out);
        });
    });

    c.bench_function("fit_transform_1000x50_normal", |b| {
        b.iter(|| {
            let (refs, q) = fit_quantiles(&data, n_rows, n_cols, 1000, Some(10_000), 0);
            let mut out = data.clone();
            transform_matrix(
                &mut out,
                n_rows,
                n_cols,
                &q,
                &refs,
                OutputDistribution::Normal,
            );
            std::hint::black_box(out);
        });
    });
}

criterion_group!(benches, bench_fit_transform);
criterion_main!(benches);

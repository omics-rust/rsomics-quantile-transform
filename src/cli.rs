use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use serde::Serialize;

use rsomics_common::{CommonFlags, Result, RsomicsError, ToolMeta, run};

use rsomics_quantile_transform::{
    Matrix, OutputDistribution, fit_quantiles, fmt_value, transform_matrix,
};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DistArg {
    /// Map to a uniform distribution in [0, 1] (default).
    Uniform,
    /// Map to a standard normal distribution.
    Normal,
}

/// Quantile transformer — value-exact scikit-learn `QuantileTransformer` port.
///
/// Reads a tab-separated `n×d` matrix (rows = samples, columns = features) from
/// a file or stdin (`-`). A leading empty top-left cell marks a header row and
/// first column holds row names. Fits per-feature empirical quantiles then maps
/// each feature to a uniform (default) or normal output distribution. Transformed
/// matrix written as TSV, or with `--json` as `{row_names, col_names, matrix}`.
#[derive(Parser, Debug)]
#[command(name = "rsomics-quantile-transform", version, about, long_about = None)]
pub struct Cli {
    /// Feature matrix TSV (`-` or omitted reads stdin).
    #[arg(value_name = "MATRIX")]
    pub matrix: Option<PathBuf>,

    /// Number of quantile landmarks (clamped to n_samples when larger).
    #[arg(long, default_value_t = 1000, value_name = "N")]
    pub n_quantiles: usize,

    /// Output distribution for each feature.
    #[arg(long, value_enum, default_value_t = DistArg::Uniform)]
    pub output_distribution: DistArg,

    /// Max samples used to estimate quantiles; 0 disables subsampling.
    #[arg(long, default_value_t = 10000, value_name = "N")]
    pub subsample: usize,

    /// Random seed for subsampling (only relevant when n_samples > --subsample).
    #[arg(long, default_value_t = 0, value_name = "SEED")]
    pub random_state: u64,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
struct MatrixOut {
    row_names: Vec<String>,
    col_names: Vec<String>,
    matrix: Vec<Vec<f64>>,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let m = Matrix::read(self.matrix.as_deref())?;
            let dist = match self.output_distribution {
                DistArg::Uniform => OutputDistribution::Uniform,
                DistArg::Normal => OutputDistribution::Normal,
            };

            if self.n_quantiles == 0 {
                return Err(RsomicsError::InvalidInput(
                    "--n-quantiles must be ≥ 1".into(),
                ));
            }

            let subsample = if self.subsample == 0 {
                None
            } else {
                Some(self.subsample)
            };

            let (references, quantiles_per_col) = fit_quantiles(
                &m.data,
                m.n_rows,
                m.n_cols,
                self.n_quantiles,
                subsample,
                self.random_state,
            );

            let mut out = m.data.clone();
            transform_matrix(
                &mut out,
                m.n_rows,
                m.n_cols,
                &quantiles_per_col,
                &references,
                dist,
            );

            if !common.json {
                let stdout = std::io::stdout().lock();
                let mut w = BufWriter::new(stdout);
                write_tsv(&mut w, &m, &out)?;
                w.flush().map_err(RsomicsError::Io)?;
            }

            Ok(MatrixOut {
                row_names: m.row_names,
                col_names: m.col_names,
                matrix: out.chunks(m.n_cols).map(<[f64]>::to_vec).collect(),
            })
        })
    }
}

fn write_tsv<W: Write>(w: &mut W, m: &Matrix, data: &[f64]) -> Result<()> {
    writeln!(w, "\t{}", m.col_names.join("\t")).map_err(RsomicsError::Io)?;
    for (i, row) in data.chunks(m.n_cols).enumerate() {
        let cells: Vec<String> = row.iter().map(|&v| fmt_value(v)).collect();
        writeln!(w, "{}\t{}", m.row_names[i], cells.join("\t")).map_err(RsomicsError::Io)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}

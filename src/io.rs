//! Dense feature-matrix I/O — tab-separated `n×d`, header auto-detected.

use std::io::Read;
use std::path::Path;

use rsomics_common::{Result, RsomicsError};

/// Row-major dense matrix with optional row/column labels.
pub struct Matrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub row_names: Vec<String>,
    pub col_names: Vec<String>,
    pub data: Vec<f64>,
}

impl Matrix {
    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.n_cols..(i + 1) * self.n_cols]
    }

    /// Materialise column `j` into a fresh `Vec` (row-major storage strides).
    #[must_use]
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.n_rows)
            .map(|i| self.data[i * self.n_cols + j])
            .collect()
    }

    /// Read from a file (`-` or `None` = stdin).
    ///
    /// # Errors
    /// Missing file, empty input, ragged body, non-numeric cell.
    pub fn read(path: Option<&Path>) -> Result<Matrix> {
        let mut buf = String::new();
        match path {
            Some(p) if p.as_os_str() != "-" => {
                std::fs::File::open(p)
                    .map_err(|e| RsomicsError::InvalidInput(format!("{}: {e}", p.display())))?
                    .read_to_string(&mut buf)
                    .map_err(RsomicsError::Io)?;
            }
            _ => {
                std::io::stdin()
                    .lock()
                    .read_to_string(&mut buf)
                    .map_err(RsomicsError::Io)?;
            }
        }
        Matrix::parse(&buf)
    }

    /// Parse tab-separated matrix. Header detected by leading tab (empty top-left).
    ///
    /// # Errors
    /// Empty input, ragged body, non-numeric cell.
    pub fn parse(text: &str) -> Result<Matrix> {
        let rows: Vec<&str> = text
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .collect();
        if rows.is_empty() {
            return Err(RsomicsError::InvalidInput("empty matrix".into()));
        }

        let has_header = rows[0].starts_with('\t');
        let (col_names, body) = if has_header {
            let names: Vec<String> = rows[0].split('\t').skip(1).map(str::to_string).collect();
            (names, &rows[1..])
        } else {
            let ncol = rows[0].split('\t').count();
            ((1..=ncol).map(|j| j.to_string()).collect(), &rows[..])
        };

        let n_cols = col_names.len();
        if n_cols == 0 {
            return Err(RsomicsError::InvalidInput("matrix has no columns".into()));
        }

        let mut row_names = Vec::with_capacity(body.len());
        let mut data = Vec::with_capacity(body.len() * n_cols);
        for (li, line) in body.iter().enumerate() {
            let mut fields = line.split('\t');
            if has_header {
                row_names.push(fields.next().unwrap_or("").to_string());
            } else {
                row_names.push((li + 1).to_string());
            }
            let mut seen = 0;
            for cell in fields {
                data.push(parse_cell(cell)?);
                seen += 1;
            }
            if seen != n_cols {
                return Err(RsomicsError::InvalidInput(format!(
                    "row {} has {seen} value columns, expected {n_cols}",
                    li + 1
                )));
            }
        }

        let n_rows = row_names.len();
        if n_rows == 0 {
            return Err(RsomicsError::InvalidInput("matrix has no data rows".into()));
        }

        Ok(Matrix {
            n_rows,
            n_cols,
            row_names,
            col_names,
            data,
        })
    }
}

fn parse_cell(cell: &str) -> Result<f64> {
    let t = cell.trim();
    match t {
        "" | "NA" | "NaN" | "na" | "nan" => Ok(f64::NAN),
        "Inf" | "inf" | "+Inf" => Ok(f64::INFINITY),
        "-Inf" | "-inf" => Ok(f64::NEG_INFINITY),
        _ => fast_float2::parse(t)
            .map_err(|_| RsomicsError::InvalidInput(format!("non-numeric cell '{cell}'"))),
    }
}

/// Shortest round-trip decimal; `NA`/`Inf`/`-Inf` for non-finite.
#[must_use]
pub fn fmt_value(v: f64) -> String {
    if v.is_nan() {
        "NA".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Inf".into() } else { "-Inf".into() }
    } else if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e16) {
        format!("{v:e}")
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_with_header() {
        let m = Matrix::parse("\tc1\tc2\tc3\nr1\t1\t2\t3\nr2\t4\t5\t6\n").unwrap();
        assert_eq!(m.n_rows, 2);
        assert_eq!(m.n_cols, 3);
        assert_eq!(m.row_names, ["r1", "r2"]);
        assert_eq!(m.col_names, ["c1", "c2", "c3"]);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(m.column(0), [1.0, 4.0]);
    }

    #[test]
    fn parse_headerless() {
        let m = Matrix::parse("1\t2\n3\t4\n").unwrap();
        assert_eq!(m.n_rows, 2);
        assert_eq!(m.n_cols, 2);
        assert_eq!(m.row(0), &[1.0, 2.0]);
        assert_eq!(m.col_names, ["1", "2"]);
    }

    #[test]
    fn ragged_row_errors() {
        assert!(Matrix::parse("1\t2\t3\n4\t5\n").is_err());
    }
}

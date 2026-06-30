pub mod io;
pub mod ndtri;
pub mod quantile;
pub mod rng;
pub mod transform;

pub use io::{Matrix, fmt_value};
pub use quantile::fit_quantiles;
pub use transform::{OutputDistribution, transform_matrix};

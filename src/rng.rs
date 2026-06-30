//! Legacy numpy `RandomState` MT19937 — the RNG sklearn's subsampling uses.
//!
//! `QuantileTransformer._dense_fit` calls `resample(X, replace=False, n_samples=k,
//! random_state=rng)` where `rng = check_random_state(seed)`. With `replace=False`,
//! `resample` does `rng.shuffle(np.arange(n)); indices[:k]`. `shuffle` on a
//! `RandomState` uses bottom-up Fisher–Yates with `rk_interval(i)` as the draw.
//!
//! Seeding: single int → Knuth `init_genrand`. Differs from numpy's `MT19937`
//! default generator (modern API) — `check_random_state(int)` builds a legacy
//! `RandomState` which is what we port here.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;

pub struct Mt19937 {
    mt: [u32; N],
    idx: usize,
}

impl Mt19937 {
    pub fn seed(seed: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for i in 1..N {
            let prev = mt[i - 1];
            mt[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: N }
    }

    fn generate(&mut self) {
        for i in 0..N {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % N] & LOWER);
            let mag = if y & 1 == 0 { 0 } else { MATRIX_A };
            self.mt[i] = self.mt[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.idx = 0;
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            self.generate();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `rk_interval(maxv)`: mask low `bit_length(maxv)` bits, accept `<= maxv`.
    #[inline]
    fn rk_interval(&mut self, maxv: u32) -> u32 {
        if maxv == 0 {
            return 0;
        }
        let mut mask = maxv;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_u32() & mask;
            if v <= maxv {
                return v;
            }
        }
    }
}

/// `np.random.RandomState(seed).shuffle(np.arange(n))` — bottom-up Fisher–Yates
/// with `rk_interval(i)` as swap partner. Returns all `n` indices in shuffled order.
///
/// `resample(X, replace=False, n_samples=k)` then takes `[:k]`.
pub fn shuffle_arange(rng: &mut Mt19937, n: usize) -> Vec<usize> {
    let mut arr: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.rk_interval(i as u32) as usize;
        arr.swap(i, j);
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_stream_seed0() {
        let mut rng = Mt19937::seed(0);
        let got: Vec<u32> = (0..8).map(|_| rng.next_u32()).collect();
        assert_eq!(
            got,
            [
                2357136044, 2546248239, 3071714933, 3626093760, 2588848963, 3684848379, 2340255427,
                3638918503
            ]
        );
    }

    #[test]
    fn shuffle_matches_numpy_seed0() {
        // np.random.RandomState(0).shuffle(np.arange(10)) verified value
        let mut rng = Mt19937::seed(0);
        let p = shuffle_arange(&mut rng, 10);
        assert_eq!(p, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]);
    }

    #[test]
    fn shuffle_matches_numpy_seed42() {
        let mut rng = Mt19937::seed(42);
        let p = shuffle_arange(&mut rng, 20);
        assert_eq!(
            p,
            [
                0, 17, 15, 1, 8, 5, 11, 3, 18, 16, 13, 2, 9, 19, 4, 12, 7, 10, 14, 6
            ]
        );
    }
}

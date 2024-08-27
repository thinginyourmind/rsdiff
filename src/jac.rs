use crate::appr_cmp::{approx_eq, approx_eq_mat};
use crate::diff::{partial_der, DiffError, DiffResult};
use nalgebra::{dmatrix, zero, DMatrix};

pub fn jac_mat(
    f_in: &impl Fn(DMatrix<f32>) -> DMatrix<f32>,
    x0: &DMatrix<f32>,
    eps: f32,
) -> DiffResult<DMatrix<f32>> {
    let n = x0.ncols();
    let mut res = DMatrix::repeat(n, n, 0.0);
    for j in 0..n {
        let f1 = |x| f_in(x)[j];
        for i in 0..n {
            res[(i, j)] = partial_der(&f1, x0, eps, i).unwrap();
        }
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(x0: DMatrix<f32>) -> DMatrix<f32> {
        let mut x1 = x0.clone();
        x1[0] = x0[0] * x0[1].powi(2);
        x1[1] = x0[1] * x0[0].powi(2);
        x1
    }

    fn target_jac(x0: DMatrix<f32>) -> DMatrix<f32> {
        let mut x1 = DMatrix::repeat(x0.ncols(), x0.ncols(), 0.0);
        x1[(0, 0)] = x0[1].powi(2);
        x1[(0, 1)] = 2.0 * x0[0] * x0[1];
        x1[(1, 0)] = 2.0 * x0[0] * x0[1];
        x1[(1, 1)] = x0[0].powi(2);
        x1
    }

    #[test]
    fn test_jac_mat() {
        let x0 = dmatrix![1.2, 3.1];
        let eps = 1e-3;
        let tol = 1e-1;
        let num_jac = jac_mat(&target, &x0, eps).unwrap();
        let a_jac = target_jac(x0);
        println!("{}", approx_eq_mat(&num_jac, &a_jac, tol).unwrap());
    }
}

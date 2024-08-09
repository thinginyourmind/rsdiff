//use std::fmt::Result;

use std::process::Output;

use crate::appr_cmp::{approx_eq, approx_eq_mat};
use nalgebra::{dmatrix, zero, DMatrix};

#[derive(Debug)]
struct DiffError;

pub type DiffResult<T> = std::result::Result<T, DiffError>;

static CFD: [f32; 5] = [1. / 12., -2. / 3., 0.0, 2. / 3., -1. / 12.];
static CFD2: [f32; 5] = [-1. / 12., 4. / 3., -5. / 2., 4. / 3., -1. / 12.];
static OFFSET: f32 = -2.0;

pub fn df_abstr(
    f_in: &impl Fn(f32) -> f32,
    x0: f32,
    eps: f32,
    coeffs: [f32; 5],
    order: i32,
) -> DiffResult<f32> {
    let mut res: f32 = 0.0;
    let mut x1: f32 = x0 - eps * OFFSET;
    for weight in coeffs {
        if weight != 0.0 {
            res += weight * f_in(x1);
        }
        x1 += eps;
    }
    Ok(res / eps.powi(order))
}

pub fn df1(f_in: &impl Fn(f32) -> f32, x0: f32, eps: f32) -> Result<f32, DiffError> {
    df_abstr(f_in, x0, eps, CFD, 1)
}

pub fn df2(f_in: &impl Fn(f32) -> f32, x0: f32, eps: f32) -> Result<f32, DiffError> {
    df_abstr(f_in, x0, eps, CFD2, 2)
}

pub fn partial_der(
    f_in: &impl Fn(DMatrix<f32>) -> f32,
    x0: &DMatrix<f32>,
    eps: f32,
    ind: usize,
) -> DiffResult<f32> {
    let fi = |x: f32| {
        let mut x1 = x0.clone();
        x1[ind] = x;
        f_in(x1)
    };
    df1(&fi, x0[ind], eps)
}

pub fn grad(
    f_in: &impl Fn(DMatrix<f32>) -> f32,
    x0: &DMatrix<f32>,
    eps: f32,
) -> DiffResult<DMatrix<f32>> {
    let mut res: DMatrix<f32> = x0.clone();
    match x0.nrows() {
        1 => {
            for i in 0..x0.ncols() {
                res[i] = partial_der(f_in, x0, eps, i).unwrap();
            }
            Ok(res)
        }
        _ => Err(DiffError),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_df_1_2() {
        let f = |x: f32| 3.0 * x.powi(3) - 1.0;
        let dfdx_1 = |x: f32| 9.0 * x.powi(2);
        let dfdx_2 = |x: f32| 18.0 * x;
        let x0 = 1.0;
        let tol = 0.15;
        let eps = 1e-3;
        println!("{}, {}", x0, dfdx_1(x0));
        println!(
            "{}",
            approx_eq(dfdx_1(x0), df1(&f, x0, eps).unwrap(), tol).unwrap()
        );
        println!("{}, {}", x0, dfdx_2(x0));
        println!(
            "{}",
            approx_eq(dfdx_2(x0), df2(&f, x0, eps).unwrap(), tol).unwrap()
        );
    }

    #[test]
    fn test_partial_der() {
        let f = |x: DMatrix<f32>| x[0].powi(3) * x[1] - x[1].powi(3) * x[0];
        let dfdx = |x: DMatrix<f32>| 3.0 * x[0].powi(2) * x[1] - x[1].powi(3);
        let dfdy = |x: DMatrix<f32>| -3.0 * x[0] * x[1].powi(2) + x[0].powi(3);
        let x0 = dmatrix![1.0, 2.0];
        let eps = 1e-3;
        let tol = 1e-1;
        let num_dfdx = partial_der(&f, &x0, eps, 0).unwrap();
        let num_dfdy = partial_der(&f, &x0, eps, 1).unwrap();
        let num_grad = grad(&f, &x0, eps).unwrap();
        println!("{}", approx_eq(num_dfdx, dfdx(x0.clone()), tol).unwrap());
        println!("{}", approx_eq(num_dfdy, dfdy(x0.clone()), tol).unwrap());
        println!(
            "{}",
            approx_eq_mat(
                &num_grad,
                &dmatrix![dfdx(x0.clone()), dfdy(x0.clone())],
                tol
            )
            .unwrap()
        );
    }
}

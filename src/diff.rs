use nalgebra::DMatrix;
use crate::appr_cmp::{approx_eq, approx_eq_mat};

static CFD: [f32; 5] = [
    1./12.,
    -2./3.,
    0.0,
    2./3.,
    -1./12.,
];
static CFD2: [f32; 5] = [ -1./12., 	4./3., 	-5./2., 	4./3., 	-1./12., ];
static OFFSET: f32 = -2.0;

pub fn df_abstr(f_in: &impl Fn(f32) -> f32, x0: f32, eps: f32, coeffs: [f32; 5], order: i32) -> f32 {
    let mut res: f32 = 0.0;
    let mut x1: f32 = x0 - eps * OFFSET;
    for weight in coeffs {
        if weight != 0.0 {
            res += weight * f_in(x1);
        }
        x1 += eps;
    }
    res / eps.powi(order)
}

pub fn df1(f_in: &impl Fn(f32) -> f32, x0: f32, eps: f32) -> f32 {
    df_abstr(f_in, x0, eps, CFD, 1)
}

pub fn df2(f_in: &impl Fn(f32) -> f32, x0: f32, eps: f32) -> f32 {
    df_abstr(f_in, x0, eps, CFD2, 2)
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
        println!("{}", approx_eq(dfdx_1(x0), df1(&f, x0, eps), tol).unwrap());
        println!("{}, {}", x0, dfdx_2(x0));
        println!("{}", approx_eq(dfdx_2(x0), df2(&f, x0, eps), tol).unwrap());
    }

}

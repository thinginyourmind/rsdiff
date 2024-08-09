use nalgebra::DMatrix;
use std::cmp::Ordering;
use std::fmt;
#[derive(Debug)]
pub struct ApproxEqError {
    d: f32,
}

impl fmt::Display for ApproxEqError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Non-equal parts")
    }
}

pub fn approx_eq(v1: f32, v2: f32, tol: f32) -> Result<f32, ApproxEqError> {
    match (v1 - v2).abs() {
        d if d <= tol => Ok(d),
        d if d > tol => Err(ApproxEqError { d }),
        _ => Err(ApproxEqError { d: 0.0 }),
    }
}

pub fn approx_eq_mat(v1: &DMatrix<f32>, v2: &DMatrix<f32>, tol: f32) -> Result<f32, ApproxEqError> {
    match v1.shape().cmp(&v2.shape()) {
        Ordering::Equal => {
            let mut total_error: f32 = 0.0;
            for i in 0..v1.shape().0 {
                for j in 0..v1.shape().1 {
                    match approx_eq(v1[(i, j)], v2[(i, j)], tol) {
                        Ok(d) => total_error += d,
                        Err(e) => return Err(e),
                    }
                }
            }
            Ok(total_error / (v1.ncols() as f32) / (v1.nrows() as f32))
        }
        _ => Err(ApproxEqError { d: 0.0 }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_eq() {
        let v1: f32 = 1.0;
        let v2: f32 = 1.01;
        let tol = 0.011;
        println!("{}", approx_eq(v1, v2, tol).unwrap());
        println!("{}", approx_eq(v2, v1, tol).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_approx_eq_false() {
        let v1: f32 = 1.0;
        let v2: f32 = 1.01;
        let tol = 0.009;
        println!("{}", approx_eq(v1, v2, tol).unwrap());
        println!("{}", approx_eq(v2, v1, tol).unwrap());
    }

    #[test]
    fn test_approx_eq_mat() {
        let v1: DMatrix<f32> = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        let v2: DMatrix<f32> = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.01, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0, 0.0,
            ],
        );
        let tol = 0.011;
        println!("{}", approx_eq_mat(&v1, &v2, tol).unwrap());
        println!("{}", approx_eq_mat(&v2, &v1, tol).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_approx_eq_mat_false() {
        let v1: DMatrix<f32> = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        let v2: DMatrix<f32> = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.01, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0, 0.0,
            ],
        );
        let tol = 0.009;
        println!("{}", approx_eq_mat(&v1, &v2, tol).unwrap());
        println!("{}", approx_eq_mat(&v2, &v1, tol).unwrap());
    }
}

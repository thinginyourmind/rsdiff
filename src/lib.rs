use nalgebra::DMatrix;
use std::fmt;

#[derive(Debug)]
pub struct ApproxEqError { tol: f32, }
type DiffWeights = [f32; 9];
type Result<T> = std::result::Result<T, ApproxEqError>;

static CFD: DiffWeights = [
    1.0 / 280.0,
    -4.0 / 105.0,
    0.2,
    -0.8,
    0.0,
    0.8,
    -0.2,
    4.0 / 105.0,
    -1.0 / 280.0,
];

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

impl fmt::Display for ApproxEqError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Non-equal parts")
    }
}

pub fn approx_eq(v1: f32, v2: f32, tol: f32) -> Result<bool> {
    match (v1-v2).abs() {
        d if d <= tol => Ok(true),
        d if d > tol => Err(ApproxEqError{tol}),
        _ => Err(ApproxEqError{tol}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

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
}

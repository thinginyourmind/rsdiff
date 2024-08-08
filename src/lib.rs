mod appr_cmp;
use nalgebra::DMatrix;

type DiffWeights = [f32; 9];
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



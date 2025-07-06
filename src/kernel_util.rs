use cubecl::{cube, prelude::*};
use crate::{NBIN, RADIUS, MODEL_TYPE};
use std::f32;

#[cube]
pub fn sign<F: Float>(x: F) -> F {
    let zero = F::cast_from(0.0f32);
    let one = F::cast_from(1.0f32);
    let neg_one = F::cast_from(-1.0f32);
    let sign = if x > zero {
        one
    } else if x < zero {
        neg_one
    } else {
        zero
    };
    sign
}

#[cube]
pub fn get_distance<F: Float>(x: F, y: F, l: F, is_signed: bool, is_periodic: bool) -> F {
    let distance: F = if is_periodic == true { 
        get_peuclidean_distance::<F>(x, y, l, is_signed)
    } else { 
        get_euclidean_distance::<F>(x, y, is_signed)
    };
    distance
}

#[cube]
pub fn get_peuclidean_distance<F: Float>(x: F, y:F, l: F, is_signed: bool) -> F {
    let s1 = F::abs(x - y);
    let s2: F = F::rem(s1, l);
    let s3 = F::min(s2, l - s2);
    let ret = F::abs(s3);
    let retval: F = if is_signed == true {
        ret * sign::<F>(l - s2 - s2) * sign::<F>(x-y)
    } else {
        ret
    };
    retval
}

#[cube]
pub fn get_euclidean_distance<F: Float>(x: F, y:F, is_signed: bool) -> F {
    let ret = x - y;
    let retval: F = if is_signed == true {
        ret
    } else {
        F::abs(ret)
    };
    retval
}

use cubecl::{cube, prelude::*};
use crate::{
    NBIN, 
    RADIUS, 
    EPSILON,
    F,
    L,
    DENSITY,
    MODEL_TYPE, 
    NP,
    DT,
    kernel_util::{*}
};

#[cube(launch)]
pub fn bin_kernel<F: Float, I: Int>(
    input: &Tensor<F>,
    output: &mut Tensor<I>
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = I::cast_from(<F as cubecl::frontend::Floor>::floor(input[ABSOLUTE_POS] * F::cast_from(NBIN) / F::cast_from(L)));
    } else {
        terminate!();
    }
}

#[cube(launch_unchecked)]
pub fn zero_init_kernel<I: Int>(
    output: &mut Tensor<I>,
    size: u32,
) {
    let global_idx = ABSOLUTE_POS;
    if global_idx < size {
        output[global_idx] = I::cast_from(0);
    } else {
        terminate!();
    }
}

#[cube(launch)]
pub fn histogram_kernel<I: Int>(
    input: &Tensor<I>,
    offset: &mut Tensor<Atomic<I>>,
    output: &mut Tensor<Atomic<I>>
) {
    Atomic::add(&output[u32::cast_from(1)], I::cast_from(1));
}


mod forward;
mod kernel;
mod kernel_util;
pub mod util;

use burn::tensor::{Tensor, TensorPrimitive, ops::{FloatTensor, IntTensor}};
use burn::prelude::Float;
use burn::prelude::Int;
use cubecl::cube;
use comptime::comptime;

#[cfg(feature = "test10")]
pub const NP: u32 = 1024;
#[cfg(feature = "test10")]
pub const NBIN: f32 = 32.0;

pub const EPSILON: f32 = f32::EPSILON;
pub const DT: f32 = 0.001;
pub const NE: f32 = 10000.0; 
pub const L: f32 = 1.0;
pub const DENSITY: f32 = 0.1;
// radius of a single particle
pub const A: f32 = L * DENSITY / ((NP as f32) * 2.0);
// the length of a single bin
pub const BINLENGTH: f32 = L / NBIN;
// how many times the interaction should be of the radius of a particle
pub const F: f32 = 5.0;
// interaction radius between particles
pub const RADIUS: f32 = F * A;
pub const MODEL_TYPE: u32 = 10;
pub const SAMPLING_RATE: f32 = 10.0;

pub trait Backend: burn::tensor::backend::Backend {
    fn call_histogram(
        input: IntTensor<Self>
    ) -> (IntTensor<Self>, IntTensor<Self>);
    fn call_bin(xs: FloatTensor<Self>) -> IntTensor<Self>;
}

pub fn bin<B: Backend, const D: usize>(
    xs: Tensor<B, D, Float>
) -> Tensor<B, D, Int> {
    let output = B::call_bin(xs.into_primitive().tensor());
    Tensor::from_primitive(output)
}

pub fn histogram<B: Backend, const D: usize>(
    input: Tensor<B, D, Int>
) -> (Tensor<B, D, Int>, Tensor<B, D, Int>) {
    let input_primitive = input.into_primitive();
    let (hist, offset) = B::call_histogram(input_primitive);
    (Tensor::from_primitive(hist), Tensor::from_primitive(offset))
}

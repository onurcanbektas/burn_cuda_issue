#[cfg(feature = "wgpu")]
use burn::backend::wgpu::WgpuRuntime;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;
use burn::tensor::Tensor;
use test1::{*, util::{*}};
use burn::prelude::Int;
use burn::prelude::Float;
use burn::tensor::Shape;
use cubecl::Runtime;
use std::time::Instant;
use core::panic;
use burn::tensor::ElementConversion;
use burn::tensor::Distribution;

fn run<B: Backend>(device: &B::Device) {
    let noise_pos = burn::tensor::Distribution::Uniform(0.0, L as f64);
    let x = Tensor::<B, 1, Float>::random(Shape::new([NP as usize]), noise_pos, device);
    println!("Initial conditions:");
    println!("x {}", x.clone());

    let bins = bin::<B, 1>(x.clone());
    let (hist, offset) = histogram::<B, 1>(bins.clone());
    B::sync(&Default::default());
}

fn main() {
    #[cfg(feature = "cuda")]
    type MyBackend = burn_cubecl::CubeBackend<CudaRuntime, f32, i32, u8>;

    #[cfg(feature = "wgpu")]
    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

    let device = Default::default();

    #[cfg(feature = "cuda")]
    run::<MyBackend>(&device);

    #[cfg(feature = "wgpu")]
    run::<MyBackend>(&device);


}

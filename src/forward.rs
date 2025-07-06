use crate::{FloatTensor, IntTensor, 
    kernel::{*}, *};
use super::Backend;
use burn::tensor::Shape;
use burn_cubecl::{
    CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement,
    kernel::into_contiguous, tensor::CubeTensor,
};
use cubecl::{CubeCount, CubeDim};
#[warn(unused_imports)]
use burn::prelude::Backend as OtherBackend;
use cubecl::prelude::SequenceArg;
use cubecl::prelude::ArrayArg;
use cubecl::CubeElement;

/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime, F: FloatElement, I: IntElement + cubecl::frontend::Index, BT: BoolElement> Backend
    for CubeBackend<R, F, I, BT>
{
    fn call_bin(
        input: FloatTensor<Self>
    ) -> IntTensor<Self> {
        // Make sure the tensor is contiguous
        let input = into_contiguous(input);
        let num_elements = input.shape.num_elements();

        let output_shape = Shape::new([num_elements]);
        let buffer_output = input
            .client
            .empty(output_shape.num_elements() * core::mem::size_of::<I>());
        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            output_shape,
            buffer_output,
            I::dtype(),
        );

        // Define cube dim and count for counting sort
        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
        let cubes_needed = f32::ceil(num_elements as f32 / cube_dim.x as f32) as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);

        // Launch atomic counting sort kernel to get histogram
        bin_kernel::launch::<F, I, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<I>(1)
        );

        output
    }

    fn call_histogram(
        input: IntTensor<Self>
    ) -> (IntTensor<Self>, IntTensor<Self>) {

        // Make sure the tensor is contiguous
        let input = into_contiguous(input);
        let max_bin = NBIN as usize;

        // Create output tensor for histogram
        // the zeroth element will be a special element
        // we are going to fill the rest as usual
        let output_shape = Shape::new([max_bin + 1]);
        let buffer_output = input
            .client
            .empty(output_shape.num_elements() * core::mem::size_of::<I>());
        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            output_shape,
            buffer_output,
            I::dtype(),
        );
        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
        let num_elements = output.shape.num_elements();
        let cubes_needed = f32::ceil(num_elements as f32 / cube_dim.x as f32) as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);
        unsafe {
            zero_init_kernel::launch_unchecked::<I, R>(
                &input.client.clone(),
                cube_count.clone(),
                cube_dim.clone(),
                output.as_tensor_arg::<I>(1),
                cubecl::frontend::ScalarArg { elem: (num_elements as u32) }
            );
        }

        let offset_shape = Shape::new([input.shape.num_elements()]);
        let buffer_offset = input
            .client
            .empty(offset_shape.num_elements() * core::mem::size_of::<I>());
        let offset = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            offset_shape,
            buffer_offset,
            I::dtype(),
        );
        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
        let num_elements = offset.shape.num_elements();
        let cubes_needed = f32::ceil(num_elements as f32 / cube_dim.x as f32) as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);
        unsafe {
            zero_init_kernel::launch_unchecked::<I, R>(
                &input.client,
                cube_count,
                cube_dim,
                offset.as_tensor_arg::<I>(1),
                cubecl::frontend::ScalarArg { elem: (num_elements as u32) }
            );
        }

        let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
        let num_elements = input.shape.num_elements();
        let cubes_needed = f32::ceil(num_elements as f32 / cube_dim.x as f32) as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);
        histogram_kernel::launch::<I, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<I>(1),
            offset.as_tensor_arg::<I>(1),
            output.as_tensor_arg::<I>(1)
        );
        // Self::sync(&Default::default());

        (output, offset)
    }

}

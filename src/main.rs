#![allow(dead_code)]
#![warn(clippy::all)]
#![doc = include_str!("../README.md")]
/*
extern crate nalgebra_glm as glm;

#[macro_use]
mod util;

mod controls;
mod depth_classify;
mod error;
mod mesh;
mod wavefrontobj;

#[cfg(feature = "glium")]
mod grender;
use std::f32::consts::TAU;

#[cfg(feature = "glium")]
use grender as renderer;

#[cfg(feature = "vulkano")]
mod vkrender;
#[cfg(feature = "vulkano")]
use vkrender as renderer;

fn main() {
    let input = std::env::args()
        .nth(1)
        .unwrap_or("./test_assets/bunny.obj".to_string());

    let obj = wavefrontobj::load(input).expect("pass a valid file path");
    // dbg!(obj.tris().collect::<Vec<_>>());

    // renderer::display_model(obj);
    // screenshots(obj);
    // screenshots_compare(obj);

    let orbit_amt = glm::vec2(0.01, 0.0);
    depth_classify::dual_render::display_duel_render(obj, orbit_amt);
}

fn screenshots(obj: mesh::MeshData) {
    let cnt = 32;
    let v = Vec::from_iter((0..cnt).map(|i| {
        let theta = i as f32 / cnt as f32;
        glm::vec3(theta.cos() * 3.0, 0.0, theta.sin() * 3.0)
    }));
    let paths = renderer::depth_screenshots(obj, (512, 512), &v);
    for path in paths {
        print!("{} ", path);
    }
    println!();
}

fn screenshots_compare(obj: mesh::MeshData) {
    let cnt = 360;
    let diff = TAU / cnt as f32;
    let v = Vec::from_iter((0..cnt).map(|i| {
        let theta = i as f32 * core::f32::consts::TAU / cnt as f32;
        [
            glm::vec3(theta.cos() * 3.0, 0.0, theta.sin() * 3.0),
            glm::vec3((theta + diff).cos() * 3.0, 0.0, (theta + diff).sin() * 3.0),
        ]
    }));
    let prmses = depth_classify::dual_render::depth_compare(obj, (1024, 1024), &v);
    print!("[");
    for prmse in prmses {
        print!("{} ", prmse);
    }
    println!("]");
}
*/
use vkfft::app::App;
use vkfft::app::LaunchParams;
use vkfft::config::Config;

use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    sys::{Flags, UnsafeCommandBufferBuilder},
    Kind,
};

use vulkano::instance::{Instance, InstanceExtensions};

use vulkano::VulkanLibrary;

use std::{error::Error, sync::Arc};

use util::{Context, SizeIterator, MatrixFormatter};

/// Transform a kernel from spatial data to frequency data
pub fn transform_kernel(
    context: &mut Context,
    coordinate_features: u32,
    batch_count: u32,
    size: &[u32; 2],
    kernel: &Arc<CpuAccessibleBuffer<[f32]>>,
) -> Result<(), Box<dyn Error>> {
    // Configure kernel FFT
    let config = Config::builder()
        .physical_device(context.physical)
        .device(context.device.clone())
        .fence(&context.fence)
        .queue(context.queue.clone())
        .buffer(kernel.clone())
        .command_pool(context.pool.clone())
        .kernel_convolution()
        .normalize()
        .coordinate_features(coordinate_features)
        .batch_count(1)
        .r2c()
        .disable_reorder_four_step()
        .dim(&size)
        .build()?;

    // Allocate a command buffer
    let primary_cmd_buffer = context.alloc_primary_cmd_buffer()?;

    // Create command buffer handle
    let builder =
    unsafe { UnsafeCommandBufferBuilder::new(&primary_cmd_buffer, Kind::primary())? };

    // Configure FFT launch parameters
    let mut params = LaunchParams::builder().command_buffer(&builder).build()?;

    // Construct FFT "Application"
    let mut app = App::new(config)?;

    // Run forward FFT
    app.forward(&mut params)?;
    // app.inverse(&mut params)?;

    // Dispatch command buffer and wait for completion
    let command_buffer = builder.build()?;
    context.submit(command_buffer)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("VkFFT version: {}", vkfft::version());

    let instance = Instance::new(library, Default::default())
        .unwrap_or_else(|err| panic!("Couldn't create instance: {:?}", err));

    let mut context = Context::new(&instance)?;

    let batch_count = 2;
    let coordinate_features = 2;
    let size = [32, 32];

    let kernel_size = batch_count * coordinate_features * 2 * (size[0] / 2 + 1) * size[1];

    let kernel = CpuAccessibleBuffer::from_iter(
        context.device.clone(),
        DEFAULT_BUFFER_USAGE,
        false,
        (0..kernel_size).map(|_| 0.0f32),
    )?;

    {
        let mut kernel_input = kernel.write()?;

        let mut range = size;
        range[0] = range[0] / 2 + 1;

        for f in 0..batch_count {
            for v in 0..coordinate_features {
                for [i, j] in SizeIterator::new(&range) {
                    println!("{} {}", i, j);
                    let _0 = 2 * i
                    + j * (size[0] + 2)
                    + v * (size[0] + 2) * size[1]
                    + f * coordinate_features * (size[0] + 2) * size[1];
                    let _1 = 2 * i
                    + 1
                    + j * (size[0] + 2)
                    + v * (size[0] + 2) * size[1]
                    + f * coordinate_features * (size[0] + 2) * size[1];
                    kernel_input[_0 as usize] = (f * coordinate_features + v + 1) as f32;
                    kernel_input[_1 as usize] = 0.0f32;
                }
            }
        }
    }

    println!("Kernel:");
    println!("{}", &MatrixFormatter::new(&size, &kernel));
    println!();


    transform_kernel(
        &mut context,
        coordinate_features,
        batch_count,
        &size,
        &kernel,
    )?;

    println!("Transformed Kernel:");
    println!("{}", &MatrixFormatter::new(&size, &kernel));
    println!();

    Ok(())
}

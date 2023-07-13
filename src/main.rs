#![allow(dead_code)]
#![warn(clippy::all)]
#![doc = include_str!("../README.md")]

extern crate nalgebra_glm as glm;

mod controls;
mod depth_classify;
mod error;
mod mesh;
mod wavefrontobj;

#[cfg(feature = "glium")]
mod grender;
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
    screenshots_compare(obj);
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
    let cnt = 1;
    let diff = 0.2;
    let v = Vec::from_iter((0..cnt).map(|i| {
        let theta = i as f32 / cnt as f32;
        [ glm::vec3(theta.cos() * 3.0, 0.0, theta.sin() * 3.0), 
            glm::vec3((theta + diff).cos() * 3.0, 0.0, (theta + diff).sin() * 3.0) ]
    }));
    let paths = depth_classify::dual_render::depth_compare(obj, (512, 512), &v);
    for path in paths {
        print!("{} ", path);
    }
    println!();
}

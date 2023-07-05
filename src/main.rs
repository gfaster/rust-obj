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
    let paths = renderer::depth_screenshots(
        obj, 
        (512, 512), 
        &[
            glm::vec3(3.0, 0.0, 0.0),
            glm::vec3(1.0, 3.0, 0.0),
        ]
    );
    for path in paths {
        print!("{} ", path);
    }
    println!("");

    // for some reason, we only get 800x600, and anything else will just get weird cropping
    // dbg!(renderer::depth_screenshots(
    //     obj,
    //     (800, 600),
    //     &[glm::vec3(3.0, 0.0, 0.0)]
    // ));
}

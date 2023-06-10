#![allow(dead_code)]
#![doc = include_str!("../README.md")]

extern crate glium;
extern crate nalgebra_glm as glm;

mod depth_classify;
mod error;
mod grender;
mod mesh;
mod wavefrontobj;

fn main() {
    let input = std::env::args()
        .nth(1)
        .unwrap_or("./test_assets/bunny.obj".to_string());

    let obj = wavefrontobj::load(input).expect("pass a valid file path");
    // dbg!(obj.tris().collect::<Vec<_>>());
    grender::display_model(obj);
}

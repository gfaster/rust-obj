#![allow(dead_code)]
#![warn(clippy::all)]
#![doc = include_str!("../README.md")]

extern crate nalgebra_glm as glm;

#[macro_use]
mod util;

mod controls;
mod error;
mod mesh;
mod partition;
mod wavefrontobj;

mod vkrender;
use vkrender as renderer;

fn main() {
    let input = std::env::args()
        .nth(1)
        .unwrap_or("./test_assets/bunny.obj".to_string());

    // dbg!(std::env::args().collect::<Vec<_>>());

    let obj = wavefrontobj::load(input).expect("pass a valid file path");
    // dbg!(obj.tris().collect::<Vec<_>>());

    renderer::display_model(obj);
    // screenshots(obj);
    // screenshots_compare(obj, std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(10));

    // let orbit_amt = glm::vec2(0.01, 0.0);
    // depth_classify::dual_render::display_duel_render(obj, orbit_amt);
}

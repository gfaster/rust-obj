#![allow(dead_code)]
#![warn(clippy::all)]
#![doc = include_str!("../README.md")]

extern crate nalgebra_glm as glm;

#[macro_use]
mod util;

mod controls;
mod depth_classify;
mod error;
mod mesh;
mod wavefrontobj;

use std::f32::consts::TAU;

#[cfg(feature = "vulkano")]
mod vkrender;
#[cfg(feature = "vulkano")]
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

fn screenshots_compare(obj: mesh::MeshData, res_power: i32) {
    let cnt = 36;
    let diff = TAU / 36 as f32;
    let v = Vec::from_iter((0..cnt).map(|i| {
        let theta = i as f32 * core::f32::consts::TAU / 36 as f32;
        [
            glm::vec3(theta.cos() * 3.0, 0.0, theta.sin() * 3.0),
            glm::vec3((theta + diff).cos() * 3.0, 0.0, (theta + diff).sin() * 3.0),
        ]
    }));
    // power of 2 resolution (10 is 1024 x 1024)

    eprintln!("taking {cnt} captures with delta {diff:0.2} radians at {0} x {0}", 1 << res_power);
    let avgs = depth_classify::dual_render::depth_compare(obj, (1 << res_power, 1 << res_power), &v);
   
    
    eprint!("[");
    for avg in &avgs {
        eprint!("{:.1} ", avg);
    }
    eprintln!("]");

    eprintln!("avg: {:.1}", avgs.iter().sum::<f32>() / avgs.len() as f32)
}

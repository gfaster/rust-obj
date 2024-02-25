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

use std::f32::consts::{TAU, PI};

#[cfg(feature = "vulkano")]
mod vkrender;
#[cfg(feature = "vulkano")]
use vkrender as renderer;

fn main() {
    // let input = std::env::args()
        // .nth(1)
        // .unwrap_or("./test_assets/bunny.obj".to_string());

    // dbg!(std::env::args().collect::<Vec<_>>());

    // let obj = wavefrontobj::load(input).expect("pass a valid file path");
    // dbg!(obj.tris().collect::<Vec<_>>());

    // renderer::display_model(obj);
    // screenshots(obj);
    // screenshots_compare(obj, std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(9));
    reads(std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(9)).unwrap();

    // let orbit_amt = glm::vec2(0.01, 0.0);
    // depth_classify::dual_render::display_duel_render(obj, orbit_amt);
}

fn reads(res_power: i32) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::read_to_string("inputs.txt")?;
    let mut outfile = std::fs::File::create("output.txt")?;
    outfile.set_len(0)?;
    for input in file.lines() {
        let Ok(obj) = wavefrontobj::load(input) else {
            eprintln!("reading {input:?} failed");
            continue;
        };
        screenshots_compare(obj, res_power, &mut outfile);
    }
    Ok(())
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

fn screenshots_compare(obj: mesh::MeshData, res_power: i32, outf: &mut impl std::io::Write) {
    let cnt = 36;
    // let cnt = 180;
    let diff = TAU / 36 as f32;
    // let v = Vec::from_iter(cmp_circle(cnt));
    let v = Vec::from_iter(cmp_fibsphere(cnt));
    // power of 2 resolution (10 is 1024 x 1024)
    eprintln!("taking {cnt} captures with delta {diff:0.2} radians at {0} x {0}", 1 << res_power);
    let name = obj.source().unwrap().to_string();
    let avgs = depth_classify::dual_render::depth_compare(obj, (1 << res_power, 1 << res_power), &v);
    write!(outf, "{name}").unwrap();
    
    // eprint!("[");
    for avg in &avgs {
        // eprint!("{:.1} ", avg);
        write!(outf, ",{avg}").unwrap()
    }
    // eprintln!("]");
    let avg = avgs.iter().sum::<f32>() / avgs.len() as f32;
    eprintln!("basic score: {avg:.1}"); 
    writeln!(outf).unwrap()
}

fn cmp_circle(cnt: usize) -> impl Iterator<Item = [glm::Vec3; 2]> {
    let diff = TAU / 36 as f32;
    (0..cnt).map(move |i| {
        let theta = i as f32 * core::f32::consts::TAU / 36 as f32;
        [
            glm::vec3(theta.cos() * 3.0, 0.0, theta.sin() * 3.0),
            glm::vec3((theta + diff).cos() * 3.0, 0.0, (theta + diff).sin() * 3.0),
        ]
    })
}

fn cmp_fibsphere(cnt: usize) -> impl Iterator<Item = [glm::Vec3; 2]> {
    // http://web.archive.org/web/20120421191837/http://www.cgafaq.info/wiki/Evenly_distributed_points_on_sphere
    let per_pnt = 13;
    let pnt_delta = TAU / per_pnt as f32;
    let diff = TAU / 36 as f32;
    let dlong = PI * (3.0 - 5.0f32.sqrt());
    let dz = 2.0 / cnt as f32;
    let mut long = 0.0f32;
    let mut z = 1.0 - dz/2.0;
    let dst = 3.0;
    (0..cnt).map(move |_| {
        let r = (1.0 - z * z).sqrt();
        let main = glm::vec3(long.cos() * r, z, long.sin() * r);
        let other = glm::vec3((long + diff).cos() * r, z, (long + diff).sin() * r);
        let out = (0..per_pnt).map(move |i| {
            let theta = i as f32 * pnt_delta;
            let other = glm::rotate_vec3(&other, theta, &main);
            [
                main * dst,
                other * dst
            ]
        });
        // let out = [
        //     glm::vec3(long.cos() * 3.0 * r, z * 3.0, long.sin() * 3.0 * r),
        //     glm::vec3((long + diff).cos() * 3.0 * r, z * 3.0, (long + diff).sin() * 3.0 * r),
        // ];
        // dbg!(&out);
        // dbg!(&out[0].magnitude());
        z -= dz;
        long -= dlong;
        out
    }).flatten()
}

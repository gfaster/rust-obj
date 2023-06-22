#![allow(dead_code)]
#![doc = include_str!("../README.md")]

extern crate glium;
extern crate nalgebra_glm as glm;

mod depth_classify;
mod error;
mod grender;
mod mesh;
mod wavefrontobj;

use image::*;

fn main() {
    let img1 = image::open("C:/Users/cubos/Pictures/rust_obj/21068/1687461714838.pgm").unwrap();
    let img2 = image::open("C:/Users/cubos/Pictures/rust_obj/21068/1687461716629.pgm").unwrap();

    let pd = depth_classify::pixel_difference(img1, img2).unwrap();
    println!("{}", pd.get_pixel(123, 458).channels()[0]);
    pd.save("C:/Users/cubos/Pictures/rust_obj/21068/pd.pgm").unwrap();
}

#![allow(dead_code)]
#![doc = include_str!("../README.md")]

extern crate glium;
extern crate nalgebra_glm as glm;

mod depth_classify;
mod error;
mod grender;
mod mesh;
mod wavefrontobj;

use image;

fn main() {
    let img1 = image::open("").unwrap();
    let img2 = image::open("").unwrap();

    let pd = depth_classify::pixel_difference(img1, img2).unwrap();
    pd.save("").unwrap();

}

#![allow(dead_code)]

extern crate glium;
extern crate nalgebra_glm as glm;

mod depth_classify;
mod error;
mod grender;
mod mesh;
mod wavefrontobj;

fn main() {
    let input = std::fs::read(
        std::env::args()
            .nth(1)
            .unwrap_or("./test_assets/bunny.obj".to_string()),
    )
    .expect("pass a valid file path");
    let obj = wavefrontobj::read_obj(&mut input.as_slice());
    // dbg!(obj.tris().collect::<Vec<_>>());
    grender::display_model(obj);
}

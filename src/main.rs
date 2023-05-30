extern crate glium;
extern crate nalgebra_glm as glm;

mod wavefrontobj;
mod mesh;
mod grender;
mod error;

fn main() {
    let input = std::fs::read(std::env::args().skip(1).next().unwrap_or("./test_assets/cube.obj".to_string())).expect("pass a valid file path");
    let obj = wavefrontobj::read_obj(&mut input.as_slice());
    // dbg!(obj.tris().collect::<Vec<_>>());
    grender::display_model(obj);
}

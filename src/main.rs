extern crate glium;

mod wavefrontobj;
mod mesh;
mod grender;
mod error;

fn main() {
    let input = std::fs::read(std::env::args().skip(1).next().expect("Pass a path of an obj file as an argument")).expect("pass a valid file path");
    let obj = wavefrontobj::read_obj(&mut input.as_slice());
    // dbg!(obj.tris().collect::<Vec<_>>());
    grender::display_model(obj);
}

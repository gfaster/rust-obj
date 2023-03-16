extern crate glium;

mod wavefrontobj;
mod mesh;
mod grender;

fn main() {
    let mut input = std::io::stdin().lock();
    let obj = wavefrontobj::read_obj(&mut input);
    grender::display_model(obj);
}

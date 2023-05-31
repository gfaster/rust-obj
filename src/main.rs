extern crate glium;
extern crate nalgebra_glm as glm;

mod error;
mod grender;
mod mesh;
mod wavefrontobj;

fn main() {
    let input = std::fs::read(
        std::env::args()
            .skip(1)
            .next()
            .unwrap_or("./test_assets/bunny.obj".to_string()),
    )
    .expect("pass a valid file path");
    let mut obj = wavefrontobj::read_obj(&mut input.as_slice());
    obj.recenter();
    // dbg!(obj.tris().collect::<Vec<_>>());
    grender::display_model(obj);
}

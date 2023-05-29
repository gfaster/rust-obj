use std::fs;

use crate::mesh;
use crate::mesh::GlVertex;
use glium::{glutin, Surface, Program, VertexBuffer, IndexBuffer, implement_vertex, uniform};

#[allow(unused_mut, unused_variables)]
pub fn display_model(m: mesh::MeshData){

    let mut event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let vertex_shader = fs::read_to_string("./shaders/vert.glsl")
        .expect("unable to open vert.glsl");
    let fragment_shader = fs::read_to_string("./shaders/frag.glsl")
        .expect("unable to open frag.glsl");

    // let vertex_shader = include_str!("../../shaders/vert.glsl");
    // let fragment_shader = include_str!("../../shaders/frag.glsl");

    let program = Program::from_source(&display, vertex_shader.as_ref(), fragment_shader.as_ref(), None).unwrap();

    let buffers: mesh::MeshDataBuffs = m.into();

    implement_vertex!(GlVertex, position, normal, tex);

    let mut uniforms = uniform! {
        model_transform_matrix: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0 , 0.0, 0.0, 1.0f32],
    ],
        projection_matrix: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0f32],
    ],
        view_pos: [0.0, 0.0, 0.0f32],
    };


    let vbuffer = VertexBuffer::new(&display, &buffers.verts).unwrap();
    let ibuffer = IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &buffers.indices).unwrap();

    event_loop.run(move |ev, _, control_flow| {

        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 1.0, 1.0);
        let _ = target.draw(&vbuffer, &ibuffer, &program, &uniforms, &Default::default());
        target.finish().unwrap();


        let next_frame_time = std::time::Instant::now() +
        std::time::Duration::from_nanos(16_666_667);

        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
        match ev {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            _ => (),
        }
    });
}


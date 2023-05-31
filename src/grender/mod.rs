use std::fs;

use crate::mesh;
use crate::mesh::GlVertex;
use glium::{glutin, implement_vertex, uniform, IndexBuffer, Program, Surface, VertexBuffer};

implement_vertex!(GlVertex, position, normal, tex);

#[allow(unused_mut, unused_variables)]
pub fn display_model(m: mesh::MeshData) {
    let mut event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let vertex_shader =
        fs::read_to_string("./shaders/vert.glsl").expect("unable to open vert.glsl");
    let fragment_shader =
        fs::read_to_string("./shaders/frag.glsl").expect("unable to open frag.glsl");

    // let vertex_shader = include_str!("../../shaders/vert.glsl");
    // let fragment_shader = include_str!("../../shaders/frag.glsl");

    let program = Program::from_source(
        &display,
        vertex_shader.as_ref(),
        fragment_shader.as_ref(),
        None,
    )
    .unwrap();

    if m.normalize_factor() == 0.0 {
        panic!("model has a normalization factor of zero - there are no vertices");
    }
    let scale: f32 = 2.0 / dbg!(m.normalize_factor());

    let center = m.centroid();

    let buffers: mesh::MeshDataBuffs = m.into();

    let aspect = display.get_framebuffer_dimensions();
    let perspective = glm::perspective(aspect.0 as f32 / aspect.1 as f32, 1.0, 0.01, 30.0);

    let transform = glm::Mat4::from([
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, scale, 0.0],
        [0.0, 0.0, 0.0, 1.0f32],
    ]);

    let normal_matrix = glm::transpose(&glm::inverse(&glm::mat4_to_mat3(&transform)));

    let (view, light_pos) = {
        let target_pos = center;
        let eye_pos = target_pos + glm::Vec3::from([0.0, 0.0, 3.0f32]);
        let up = glm::Vec3::from([0.0, 1.0, 0.0f32]);
        let light_pos = eye_pos + glm::Vec3::from([2.0, 2.0, 0.0f32]);
        (glm::look_at(&eye_pos, &target_pos, &up), light_pos)
    };
    // let light_pos = glm::Vec4::from([1.0, 1.0, 0.0, 1.0]);
    eprintln!("{:.2}", transform);

    let modelview = view * transform;

    for v in &buffers.verts {
        // let pos = v.position;
        // let pos4 = [pos[0], pos[1], pos[2], 1.0f32];
        // eprintln!("vpos: {}", scale * (glm::Mat4::from(perspective) *  glm::Mat4::from(transform) * glm::Vec4::from(pos4)));

        // eprintln!("vnormal: {}", glm::Vec3::from(v.normal));
    }

    let uniforms = uniform! {
        modelview: *AsRef::<[[f32; 4]; 4]>::as_ref(&modelview),
        transform: *AsRef::<[[f32; 4]; 4]>::as_ref(&transform),
        normal_matrix: *AsRef::<[[f32; 3]; 3]>::as_ref(&normal_matrix),
        projection_matrix: *AsRef::<[[f32; 4]; 4]>::as_ref(&perspective),
        light_pos: *AsRef::<[f32; 3]>::as_ref(&light_pos),
    };

    let params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let vbuffer = VertexBuffer::new(&display, &buffers.verts).unwrap();
    let ibuffer = IndexBuffer::new(
        &display,
        glium::index::PrimitiveType::TrianglesList,
        &buffers.indices,
    )
    .unwrap();

    event_loop.run(move |ev, _, control_flow| {
        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 1.0, 1.0), 1.0);
        target
            .draw(&vbuffer, &ibuffer, &program, &uniforms, &params)
            .unwrap();
        target.finish().unwrap();

        let next_frame_time =
            std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);

        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
        match ev {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                }
                _ => return,
            },
            _ => (),
        }
    });
}

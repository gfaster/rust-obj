use std::fs;
use std::time;

use crate::mesh;
use crate::mesh::GlVertex;
use glium::{
    glutin::{self, event::ElementState, window::CursorGrabMode},
    implement_vertex,
    program::ShaderStage,
    uniform, DrawParameters, IndexBuffer, Program, Surface, VertexBuffer,
};
use std::io::Write;

use self::controls::Camera;

mod controls;

pub mod consts {
    use nalgebra::ArrayStorage;
    pub const FORWARD: glm::Vec3 =
        glm::Vec3::from_array_storage(ArrayStorage([[0.0, 0.0, -1.0f32]]));
    pub const BACKWARD: glm::Vec3 =
        glm::Vec3::from_array_storage(ArrayStorage([[0.0, 0.0, 1.0f32]]));
    pub const UP: glm::Vec3 = glm::Vec3::from_array_storage(ArrayStorage([[0.0, 1.0, 0.0f32]]));
    pub const DOWN: glm::Vec3 = glm::Vec3::from_array_storage(ArrayStorage([[0.0, -1.0, 0.0f32]]));
    pub const RIGHT: glm::Vec3 = glm::Vec3::from_array_storage(ArrayStorage([[1.0, 0.0, 0.0f32]]));
    pub const LEFT: glm::Vec3 = glm::Vec3::from_array_storage(ArrayStorage([[-1.0, 0.0, 0.0f32]]));
}

#[derive(PartialEq, Eq)]
enum DrawMode {
    DepthBuffer,
    Render,
    Wire,
}

impl DrawMode {
    fn clear_color(&self) -> (f32, f32, f32, f32) {
        match self {
            DrawMode::DepthBuffer => (1.0, 1.0, 1.0, 1.0),
            DrawMode::Render | DrawMode::Wire => (0.3, 0.3, 0.5, 1.0),
        }
    }
}

enum FragSubroutine {
    Shaded,
    DepthBuffer,
}

impl FragSubroutine {
    fn as_str(&self) -> &'static str {
        match self {
            FragSubroutine::Shaded => "shaded",
            FragSubroutine::DepthBuffer => "depth_buffer",
            // FragSubroutine::Unshaded => "unshaded",
        }
    }
}

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
    let near_plane: f32 = 1.0;
    let far_plane: f32 = 4.0;
    let perspective = glm::perspective(
        aspect.0 as f32 / aspect.1 as f32,
        1.0,
        near_plane,
        far_plane,
    );

    let transform = glm::Mat4::from([
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, scale, 0.0],
        [0.0, 0.0, 0.0, 1.0f32],
    ]);
    // eprintln!("{:.2}", transform);
    let model_normal_matrix = glm::transpose(&glm::inverse(&glm::mat4_to_mat3(&transform)));

    for v in &buffers.verts {
        // let pos = v.position;
        // let pos4 = [pos[0], pos[1], pos[2], 1.0f32];
        // eprintln!("vpos: {}", scale * (glm::Mat4::from(perspective) *  glm::Mat4::from(transform) * glm::Vec4::from(pos4)));

        // eprintln!("vnormal: {}", glm::Vec3::from(v.normal));
    }

    let mut params = glium::DrawParameters {
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

    let mut camera = controls::Camera::new();
    let mut mode = DrawMode::Render;
    let mut shader_subroutine = FragSubroutine::Shaded;

    event_loop.run(move |ev, _, control_flow| {
        let view = camera.get_transform();
        let modelview = view * transform;
        let light_pos = camera.pos + glm::Vec3::from([2.0, 2.0, 0.0f32]);

        let uniforms = uniform! {
            cam_transform: *AsRef::<[[f32; 4]; 4]>::as_ref(&view),
            modelview: *AsRef::<[[f32; 4]; 4]>::as_ref(&modelview),
            transform: *AsRef::<[[f32; 4]; 4]>::as_ref(&transform),
            normal_matrix: *AsRef::<[[f32; 3]; 3]>::as_ref(&model_normal_matrix),
            projection_matrix: *AsRef::<[[f32; 4]; 4]>::as_ref(&perspective),
            light_pos: *AsRef::<[f32; 3]>::as_ref(&light_pos),
            shading_routine: (shader_subroutine.as_str(), ShaderStage::Fragment)
        };

        let mut target = display.draw();
        target.clear_color_and_depth(mode.clear_color(), 1.0);
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
                }
                glutin::event::WindowEvent::Focused(b) => {
                    handle_window_focus(b, &display);
                }
                _ => (),
            },
            glutin::event::Event::DeviceEvent { event, .. } => match event {
                glutin::event::DeviceEvent::MouseMotion { delta } => {
                    // yes, this is swapped
                    controls::mouse_move(&mut camera, &(-delta.1 as f32, -delta.0 as f32));
                }
                glutin::event::DeviceEvent::Key(k) => {
                    if k.state == ElementState::Pressed {
                        if let Some(vk) = k.virtual_keycode {
                            match vk {
                                glutin::event::VirtualKeyCode::Q => {
                                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                                }
                                glutin::event::VirtualKeyCode::W => change_draw_mode(
                                    &mut mode,
                                    DrawMode::Wire,
                                    &mut shader_subroutine,
                                    &mut params,
                                ),
                                glutin::event::VirtualKeyCode::R => change_draw_mode(
                                    &mut mode,
                                    DrawMode::Render,
                                    &mut shader_subroutine,
                                    &mut params,
                                ),
                                glutin::event::VirtualKeyCode::D => change_draw_mode(
                                    &mut mode,
                                    DrawMode::DepthBuffer,
                                    &mut shader_subroutine,
                                    &mut params,
                                ),
                                glutin::event::VirtualKeyCode::S => {
                                    match save_screenshot(
                                        &display,
                                        &camera,
                                        (near_plane, far_plane),
                                    ) {
                                        Ok(p) => eprintln!("Saved screenshot to {:?}", p),
                                        Err(e) => eprintln!("{}", e),
                                    };
                                }
                                glutin::event::VirtualKeyCode::G => {
                                    match save_screenshot_greyscale(
                                        &display,
                                        &camera,
                                        (near_plane, far_plane),
                                    ) {
                                        Ok(p) => eprintln!("Saved greyscale screenshot to {:?}", p),
                                        Err(e) => eprintln!("{}", e),
                                    };
                                }
                                _ => (),
                            }
                        }
                    }
                }
                _ => (),
            },
            _ => (),
        }
    });
}

fn handle_window_focus(focused: bool, display: &glium::Display) {
    if focused {
        display
            .gl_window()
            .window()
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_e| {
                display
                    .gl_window()
                    .window()
                    .set_cursor_grab(CursorGrabMode::Locked)
            })
            .unwrap();
        display.gl_window().window().set_cursor_visible(false);
    } else {
        display
            .gl_window()
            .window()
            .set_cursor_grab(CursorGrabMode::None)
            .unwrap();
        display.gl_window().window().set_cursor_visible(true);
    }
}

fn change_draw_mode(
    current: &mut DrawMode,
    new: DrawMode,
    shader_param: &mut FragSubroutine,
    params: &mut DrawParameters,
) {
    match (&current, &new) {
        (DrawMode::DepthBuffer, DrawMode::Render) => {
            *shader_param = FragSubroutine::Shaded;
        }
        (DrawMode::DepthBuffer, DrawMode::Wire) => {
            *shader_param = FragSubroutine::Shaded;
            params.polygon_mode = glium::PolygonMode::Line;
        }
        (DrawMode::Render, DrawMode::DepthBuffer) => {
            *shader_param = FragSubroutine::DepthBuffer;
        }
        (DrawMode::Render, DrawMode::Wire) => {
            params.polygon_mode = glium::PolygonMode::Line;
        }
        (DrawMode::Wire, DrawMode::DepthBuffer) => {
            params.polygon_mode = glium::PolygonMode::Fill;
            *shader_param = FragSubroutine::DepthBuffer;
        }
        (DrawMode::Wire, DrawMode::Render) => {
            params.polygon_mode = glium::PolygonMode::Fill;
        }

        _ => (), // identity
    }
    *current = new;
}

fn screenshot_dir() -> Result<String, Box<dyn std::error::Error>> {
    let dir_path = format!("{}/Pictures/rust_obj", std::env::var("HOME")?);
    let base_path = format!(
        "{}/Pictures/rust_obj/{}",
        std::env::var("HOME")?,
        std::process::id()
    );

    fs::create_dir(dir_path).map_or_else(
        |e| {
            if matches!(e.kind(), std::io::ErrorKind::AlreadyExists) {
                Ok(())
            } else {
                Err(e)
            }
        },
        |_| Ok(()),
    )?;
    fs::create_dir(&base_path).map_or_else(
        |e| {
            if matches!(e.kind(), std::io::ErrorKind::AlreadyExists) {
                Ok(())
            } else {
                Err(e)
            }
        },
        |_| Ok(()),
    )?;
    Ok(base_path)
}

fn save_screenshot(
    display: &glium::Display,
    cam: &Camera,
    near_far_planes: (f32, f32),
) -> Result<String, Box<dyn std::error::Error>> {
    let base_path = screenshot_dir()?;
    let name = format!(
        "{}.ppm",
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    let full_path = format!("{}/{}", &base_path, &name);
    let mut file = fs::File::options()
        .write(true)
        .create_new(true)
        .open(&full_path)?;
    let mut buffer = Vec::<u8>::new();

    let img: Vec<Vec<(u8, u8, u8, u8)>> = display.read_front_buffer()?;
    let dim = display.get_framebuffer_dimensions();

    writeln!(buffer, "P3")?;
    writeln!(
        buffer,
        "# Camera position: {}, {}, {}",
        cam.pos.x, cam.pos.y, cam.pos.z
    )?;
    writeln!(buffer, "# Camera near plane: {}", near_far_planes.0)?;
    writeln!(buffer, "# Camera far plane: {}", near_far_planes.1)?;
    writeln!(buffer, "{}, {}", dim.0, dim.1)?;
    writeln!(buffer, "255")?;

    for row in img.iter().rev() {
        for pix in row {
            write!(buffer, "{} {} {} ", pix.0, pix.1, pix.2)?;
        }
        writeln!(buffer)?;

        if buffer.len() >= 1024 * 1024 * 8 {
            file.write_all(&buffer)?;
            buffer.clear()
        }
    }
    file.write_all(&buffer)?;

    Ok(full_path)
}

fn save_screenshot_greyscale(
    display: &glium::Display,
    cam: &Camera,
    near_far_planes: (f32, f32),
) -> Result<String, Box<dyn std::error::Error>> {
    let base_path = screenshot_dir()?;
    let name = format!(
        "{}.pgm",
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    let full_path = format!("{}/{}", &base_path, &name);
    let mut file = fs::File::options()
        .write(true)
        .create_new(true)
        .open(&full_path)?;
    let mut buffer = Vec::<u8>::new();

    let img: Vec<Vec<(u8, u8, u8, u8)>> = display.read_front_buffer()?;
    let dim = display.get_framebuffer_dimensions();

    writeln!(buffer, "P2")?;
    writeln!(
        buffer,
        "# Camera position: {}, {}, {}",
        cam.pos.x, cam.pos.y, cam.pos.z
    )?;
    writeln!(buffer, "# Camera near plane: {}", near_far_planes.0)?;
    writeln!(buffer, "# Camera far plane: {}", near_far_planes.1)?;
    writeln!(buffer, "{}, {}", dim.0, dim.1)?;
    writeln!(buffer, "255")?;

    for row in img.iter().rev() {
        for pix in row {
            write!(
                buffer,
                "{} ",
                (pix.0 as u32 + pix.1 as u32 + pix.2 as u32) / 3
            )?;
        }
        writeln!(buffer)?;

        if buffer.len() >= 1024 * 1024 * 8 {
            file.write_all(&buffer)?;
            buffer.clear()
        }
    }
    file.write_all(&buffer)?;

    Ok(full_path)
}

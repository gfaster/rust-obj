extern crate glium;

use std::fs;
use std::ops::Deref;
use std::time;

use crate::mesh;
use crate::mesh::mat::Material;
use crate::mesh::GlVertex;
use crate::mesh::MeshData;
use glium::glutin::error::ExternalError;
use glium::glutin::platform::unix::HeadlessContextExt;
use glium::texture::ClientFormat;
use glium::texture::RawImage2d;
use glium::CapabilitiesSource;
use glium::{
    glutin::{self, event::ElementState, window::CursorGrabMode},
    implement_vertex,
    program::ShaderStage,
    texture::CompressedSrgbTexture2d,
    uniform, DrawParameters, IndexBuffer, Program, Surface, VertexBuffer,
};
use image::GrayImage;
use nalgebra_glm::Vec3;

use crate::controls::Camera;

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
    dbg!(display.get_capabilities());

    let material = m.material().clone();

    let vertex_shader =
        fs::read_to_string("./shaders/vert.glsl").expect("unable to open vert.glsl");
    let fragment_shader = if let Some(texture) = material.diffuse_map() {
        fs::read_to_string("./shaders/frag_textured.glsl")
            .expect("unable to open frag_textured.glsl")
    } else {
        fs::read_to_string("./shaders/frag.glsl").expect("unable to open frag.glsl")
    };

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
    let scale: f32 = 1.5 / dbg!(m.normalize_factor());

    let center = m.centroid();

    dbg!(&material);
    let gl_tex = if let Some(img) = material.diffuse_map() {
        let dim = img.dimensions();
        let img = glium::texture::RawImage2d::from_raw_rgba_reversed(&img.clone().into_raw(), dim);
        CompressedSrgbTexture2d::new(&display, img).unwrap()
    } else {
        let img = Material::dev_texture();
        let dim = img.dimensions();
        let img = glium::texture::RawImage2d::from_raw_rgba_reversed(&img.into_raw(), dim);
        CompressedSrgbTexture2d::new(&display, img).unwrap()
    };

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
    ]) * glm::Mat4::from([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-center.x, -center.y, -center.z, 1.0],
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
        // let light_pos = camera.pos + glm::Vec3::from([2.0, 2.0, 0.0f32]);
        let light_pos = camera.pos;

        let uniforms = uniform! {
            cam_transform: *AsRef::<[[f32; 4]; 4]>::as_ref(&view),
            modelview: *AsRef::<[[f32; 4]; 4]>::as_ref(&modelview),
            transform: *AsRef::<[[f32; 4]; 4]>::as_ref(&transform),
            normal_matrix: *AsRef::<[[f32; 3]; 3]>::as_ref(&model_normal_matrix),
            projection_matrix: *AsRef::<[[f32; 4]; 4]>::as_ref(&perspective),
            light_pos: *AsRef::<[f32; 3]>::as_ref(&light_pos),
            shading_routine: (shader_subroutine.as_str(), ShaderStage::Fragment),
            base_diffuse: Into::<[f32; 4]>::into(material.diffuse()),
            base_ambient: Into::<[f32; 4]>::into(material.ambient()),
            base_specular: Into::<[f32; 4]>::into(material.specular()),
            base_specular_factor: material.base_specular_factor(),
            diffuse_map: &gl_tex,
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
                    handle_window_focus(b, &display).unwrap_or(());
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

// #[allow(unused_mut, unused_variables)]
pub fn depth_screenshots(m: MeshData, dim: (u32, u32), pos: &[Vec3]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(pos.len());

    let ctx = glutin::ContextBuilder::new()
        .with_gl(glutin::GlRequest::Latest)
        .with_gl_profile(glutin::GlProfile::Core)
        .build_osmesa(dim.into())
        .unwrap();
    let facade = glium::backend::glutin::headless::Headless::new(ctx).unwrap();

    dbg!(facade.get_capabilities());
    dbg!(facade.get_opengl_version());
    dbg!(facade.get_opengl_profile());
    dbg!(facade.get_framebuffer_dimensions());

    let vertex_shader =
        fs::read_to_string("./shaders/vert.glsl").expect("unable to open vert.glsl");
    let fragment_shader =
        fs::read_to_string("./shaders/frag.glsl").expect("unable to open frag.glsl");

    // let vertex_shader = include_str!("../../shaders/vert.glsl"); let fragment_shader =
    // include_str!("../../shaders/frag.glsl");

    let program = Program::from_source(
        &facade,
        vertex_shader.as_ref(),
        fragment_shader.as_ref(),
        None,
    )
    .unwrap();

    if m.normalize_factor() == 0.0 {
        panic!("model has a normalization factor of zero - there are no vertices");
    }
    let scale: f32 = 1.5 / dbg!(m.normalize_factor());

    let center = m.centroid();

    let buffers: mesh::MeshDataBuffs = m.into();

    let aspect = facade.get_framebuffer_dimensions();
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
    ]) * glm::Mat4::from([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-center.x, -center.y, -center.z, 1.0],
    ]);

    // eprintln!("{:.2}", transform);
    let model_normal_matrix = glm::transpose(&glm::inverse(&glm::mat4_to_mat3(&transform)));

    for _v in &buffers.verts {
        // let pos = v.position; let pos4 = [pos[0], pos[1], pos[2], 1.0f32]; eprintln!("vpos: {}",
        // scale * (glm::Mat4::from(perspective) *  glm::Mat4::from(transform) *
        // glm::Vec4::from(pos4)));
    }

    // eprintln!("vnormal: {}", glm::Vec3::from(v.normal)); }

    let params = glium::DrawParameters {
        depth: glium::Depth {
            // test: glium::draw_parameters::DepthTest::IfLess,
            // write: true,
            ..Default::default()
        },
        stencil: glium::draw_parameters::Stencil {
            test_clockwise: glium::StencilTest::AlwaysPass,
            test_counter_clockwise: glium::StencilTest::AlwaysPass,
            ..Default::default()
        },
        ..Default::default()
    };

    let vbuffer = VertexBuffer::new(&facade, &buffers.verts).unwrap();
    let ibuffer = IndexBuffer::new(
        &facade,
        glium::index::PrimitiveType::TrianglesList,
        &buffers.indices,
    )
    .unwrap();

    let mut camera = controls::Camera::new();
    let mode = DrawMode::Render;
    let shader_subroutine = FragSubroutine::DepthBuffer;
    let light_pos = camera.pos;
    // let light_pos = camera.pos + glm::Vec3::from([2.0, 2.0, 0.0f32]);

    for cam_pos in pos {
        camera.set_rel_pos(*cam_pos);
        let view = camera.get_transform();
        let modelview = view * transform;

        let uniforms = uniform! {
            cam_transform: *AsRef::<[[f32; 4]; 4]>::as_ref(&view),
            modelview: *AsRef::<[[f32; 4]; 4]>::as_ref(&modelview),
            transform: *AsRef::<[[f32; 4]; 4]>::as_ref(&transform),
            normal_matrix: *AsRef::<[[f32; 3]; 3]>::as_ref(&model_normal_matrix),
            projection_matrix: *AsRef::<[[f32; 4]; 4]>::as_ref(&perspective),
            light_pos: *AsRef::<[f32; 3]>::as_ref(&light_pos),
            shading_routine: (shader_subroutine.as_str(), ShaderStage::Fragment),
            base_diffuse: [0.0_f32,0.0_f32,0.0_f32,0.0_f32],
            base_ambient: [0.0_f32,0.0_f32,0.0_f32,0.0_f32],
            base_specular: [0.0_f32,0.0_f32,0.0_f32,0.0_f32],
            base_specular_factor: 0.0_f32,
        };

        let mut target = facade.draw();
        // target.clear_color_and_depth(mode.clear_color(), 1.0);
        target.clear_all(mode.clear_color(), 1.0, 0xFF);
        dbg!(target.has_depth_buffer());
        target
            .draw(&vbuffer, &ibuffer, &program, &uniforms, &params)
            .unwrap();
        target.finish().unwrap();
        let img = capture_screen(&facade).unwrap();

        let path = "/home/gfaster/Pictures/rust_obj/tmp.png";
        img.save(path).unwrap();
        out.push(path.into());
    }
    out
}

fn handle_window_focus(focused: bool, display: &glium::Display) -> Result<(), ExternalError> {
    if focused {
        display
            .gl_window()
            .window()
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| {
                display
                    .gl_window()
                    .window()
                    .set_cursor_grab(CursorGrabMode::Locked)
            })?;
        display.gl_window().window().set_cursor_visible(false);
    } else {
        display
            .gl_window()
            .window()
            .set_cursor_grab(CursorGrabMode::None)?;
        display.gl_window().window().set_cursor_visible(true);
    }

    Ok(())
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

fn capture_screen(
    surface: &impl Deref<Target = glium::backend::Context>,
) -> Result<image::RgbaImage, Box<dyn std::error::Error>> {
    let dim = dbg!(surface.get_framebuffer_dimensions());
    let raw_img = surface.read_front_buffer::<RawImage2d<u8>>()?;
    assert!(
        matches!(raw_img.format, ClientFormat::U8U8U8U8),
        "To save an image, we assume rgba8 image format"
    );
    let mut img = image::RgbaImage::from_raw(dim.0, dim.1, raw_img.data.to_vec()).unwrap();
    image::imageops::flip_vertical_in_place(&mut img);
    Ok(img)
}

fn save_screenshot(
    display: &glium::Display,
    _cam: &Camera,
    _near_far_planes: (f32, f32),
) -> Result<String, Box<dyn std::error::Error>> {
    let base_path = screenshot_dir()?;
    let name = format!(
        "{}.png",
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    let full_path = format!("{}/{}", &base_path, &name);

    let img = capture_screen(display)?;
    img.save(full_path.clone())?;

    Ok(full_path)
}

fn save_screenshot_greyscale(
    display: &glium::Display,
    _cam: &Camera,
    _near_far_planes: (f32, f32),
) -> Result<String, Box<dyn std::error::Error>> {
    let base_path = screenshot_dir()?;
    let name = format!(
        "{}.png",
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    let full_path = format!("{}/{}", &base_path, &name);

    let dim = display.get_framebuffer_dimensions();
    let raw_img = display.read_front_buffer::<RawImage2d<u8>>()?;
    assert!(
        matches!(raw_img.format, ClientFormat::U8U8U8U8),
        "To save an image, we assume rgba8 image format"
    );
    let mut img: GrayImage = image::DynamicImage::from(
        image::RgbaImage::from_raw(dim.0, dim.1, raw_img.data.to_vec()).unwrap(),
    )
    .into_luma8();

    image::imageops::flip_vertical_in_place(&mut img);
    img.save(full_path.clone())?;

    Ok(full_path)
}

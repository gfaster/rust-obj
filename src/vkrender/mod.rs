pub mod init;
pub mod render_systems;

extern crate vulkano_shaders;

use std::collections::VecDeque;
use std::io::Write;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};

use vulkano::device::DeviceExtensions;
use vulkano::format::Format;
use vulkano::image::view::ImageView;

use vulkano::image::{Image, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};

use vulkano::pipeline::graphics::vertex_input;
use vulkano::pipeline::graphics::viewport::Viewport;

use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};

use vulkano::swapchain::{
    acquire_next_image, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{render_pass, sync, VulkanError};

use winit::event::{DeviceEvent, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;

use crate::controls::{mouse_move, Camera};

use crate::mesh::{self, MeshDataBuffs};
use render_systems::object_system::ObjectSystem;

use self::render_systems::ui_system::UiSystem;

pub mod consts {
    #![allow(unused)]

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

// format attributes use [`vulkano::format::Format`] enum fields
#[derive(BufferContents, vertex_input::Vertex, Clone, Copy)]
#[repr(C)]
pub struct VkVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],

    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],

    #[format(R32G32_SFLOAT)]
    tex: [f32; 2],
}

#[derive(BufferContents, vertex_input::Vertex, Clone, Copy)]
#[repr(C)]
pub struct PartLabel {
    #[format(R32_UINT)]
    part: u32,
}

impl From<mesh::Vertex> for VkVertex {
    fn from(value: mesh::Vertex) -> Self {
        Self {
            position: value.pos.into(),
            normal: value.normal.into(),
            tex: value.tex.into(),
        }
    }
}

impl MeshDataBuffs<VkVertex> {
    pub fn to_buffers(self, allocator: Arc<dyn MemoryAllocator>) -> Subbuffer<[VkVertex]> {
        let vertex_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.verts,
        )
        .unwrap();
        vertex_buffer
    }
}

pub fn display_model(m: mesh::MeshData) {
    // I'm using the Vulkano examples to learn here
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/triangle.rs

    let (device, queue, surface, event_loop) = init::initialize_device_window(DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    });

    let (mut swapchain, images) = {
        let surface_capabilites = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            vulkano::swapchain::SwapchainCreateInfo {
                min_image_count: surface_capabilites.min_image_count,

                image_format,

                // there is some funny driver stuff here, this is the safest option
                image_extent: window.inner_size().into(),

                image_usage: ImageUsage::COLOR_ATTACHMENT,

                composite_alpha: surface_capabilites
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            },
        )
        .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let mut cam = Camera::new(1.0);
    let mut frame_nr: usize = 0;
    let mut last_many_frames: VecDeque<std::time::Instant> = Default::default();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth: {
                format: vulkano::format::Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            }
        },
        pass: {
            color: [color],

            depth_stencil: {depth},
        }
    )
    .unwrap();

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            primary_buffer_count: 32,
            secondary_buffer_count: 32,
            ..Default::default()
        },
    ));

    let extent = {
        let dim = swapchain.image_extent();
        [dim[0] as f32, dim[1] as f32]
    };

    // TODO: creating an object system creates the pipeline and the initialize_based_on_window call
    // regenerates it
    let mut object_system = ObjectSystem::new(
        queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
        extent,
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    let mut ui_system = UiSystem::new(
        queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
        extent,
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    object_system.register_object(m, glm::Mat4::identity(), true);

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    let mut framebuffers = initialize_based_on_window(
        memory_allocator.clone(),
        &images,
        render_pass.clone(),
        &mut object_system,
        &mut ui_system,
        &mut viewport,
        &mut cam,
    );

    // initialization done!

    // swapchain can be spuriously invalidated (window resized)
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::Q),
                        ..
                    }),
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                mouse_move(&mut cam, &delta);
            }
            Event::MainEventsCleared => {
                // do not draw if size is zero (eg minimized)
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                last_many_frames.push_front(std::time::Instant::now());
                while last_many_frames.len() >= 500 {
                    last_many_frames.pop_back();
                }

                // will cause a memory leak if not called every once in a while
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            // can happen when resizing -> easy way to fix is just restart loop
                            // Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {e}"),
                        };

                    swapchain = new_swapchain;

                    let new_framebuffers = initialize_based_on_window(
                        memory_allocator.clone(),
                        &new_images,
                        render_pass.clone(),
                        &mut object_system,
                        &mut ui_system,
                        &mut viewport,
                        &mut cam,
                    );
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                // need to acquire image before drawing, blocks if it's not ready yet (too many
                // commands), so it has optional timeout
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(vulkano::Validated::Error(VulkanError::OutOfDate)) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                // sometimes this just happens, usually it's the user's fault. The image will still
                // display, just poorly. It may become an out of date error if we don't regen.
                if suboptimal {
                    recreate_swapchain = true;
                }

                ui_system.clear();
                writeln!(ui_system, "Frame nr: {}", frame_nr).unwrap();
                frame_nr += 1;
                writeln!(ui_system, "FPS: {:.1}", {
                    (last_many_frames.len() - 1) as f32
                        / last_many_frames
                            .front()
                            .unwrap()
                            .duration_since(*last_many_frames.back().unwrap())
                            .as_secs_f32()
                })
                .unwrap();
                writeln!(ui_system, "Debug: {:#?}", viewport).unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // we are allowed to not do a render pass if we use dynamic
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), Some(1.0.into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::SecondaryCommandBuffers,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .execute_commands(object_system.draw(&cam))
                    .unwrap()
                    .execute_commands(ui_system.draw())
                    .unwrap();

                // additional passes would go here
                builder.end_render_pass(SubpassEndInfo::default()).unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(f) => {
                        previous_frame_end = Some(f.boxed());
                    }
                    Err(vulkano::Validated::Error(VulkanError::OutOfDate)) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("Failed to flush future {e}");
                    }
                }
            }
            _ => (),
        }
    });
}

fn initialize_based_on_window(
    memory_allocator: Arc<dyn MemoryAllocator>,
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    object_system: &mut ObjectSystem,
    ui_system: &mut UiSystem,
    viewport: &mut Viewport,
    cam: &mut Camera,
) -> Vec<Arc<Framebuffer>> {
    let extent: [u32; 3] = images[0].extent();
    let dimensions = [extent[0] as f32, extent[1] as f32];

    *viewport = Viewport {
        offset: [0.0, 0.0],
        extent: dimensions,
        depth_range: 0.0..=1.0,
    };

    cam.aspect = dimensions[0] / dimensions[1];

    object_system.regenerate(dimensions);
    ui_system.regenerate(dimensions);

    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator,
            vulkano::image::ImageCreateInfo {
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                format: Format::D16_UNORM,
                extent,
                image_type: ImageType::Dim2d,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                render_pass::FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    framebuffers
}

pub fn screenshot_dir() -> Result<String, Box<dyn std::error::Error>> {
    let dir_path = format!("{}/Pictures/rust_obj", std::env::var("HOME")?);
    let base_path = format!(
        "{}/Pictures/rust_obj/{}",
        std::env::var("HOME")?,
        std::process::id()
    );

    std::fs::create_dir(dir_path).map_or_else(
        |e| {
            if matches!(e.kind(), std::io::ErrorKind::AlreadyExists) {
                Ok(())
            } else {
                Err(e)
            }
        },
        |_| Ok(()),
    )?;
    std::fs::create_dir(&base_path).map_or_else(
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

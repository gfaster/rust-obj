pub mod init;
pub mod render_systems;

extern crate vulkano_shaders;

use std::collections::VecDeque;
use std::io::Write;
use std::sync::Arc;

use image::Rgba32FImage;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use vulkano::device::DeviceExtensions;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, StorageImage, SwapchainImage};

use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryUsage, StandardMemoryAllocator,
};

use vulkano::pipeline::graphics::vertex_input;
use vulkano::pipeline::graphics::viewport::Viewport;

use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};

use vulkano::swapchain::{
    acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::{render_pass, sync};

use winit::event::{DeviceEvent, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Window;

use crate::controls::{mouse_move, Camera};
use crate::glm::Vec3;

use crate::mesh::{self, MeshData, MeshDataBuffs};
use crate::vkrender::init::initialize_device;
use render_systems::object_system::ObjectSystem;

use self::render_systems::ui_system::UiSystem;

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
    /// makes vertex and index subbuffers
    pub fn to_buffers(
        self,
        allocator: &impl MemoryAllocator,
    ) -> (Subbuffer<[VkVertex]>, Subbuffer<[u32]>) {
        let vertex_buffer = Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            self.verts,
        )
        .unwrap();
        let index_buffer = Buffer::from_iter(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            self.indices,
        )
        .unwrap();
        (vertex_buffer, index_buffer)
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

        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

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
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },

            depth: {
                load: Clear,
                store: DontCare,
                format: vulkano::format::Format::D16_UNORM,
                samples: 1
            }
        },
        pass: {
            color: [color],

            depth_stencil: {depth},
        }
    )
    .unwrap();

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // TODO: creating an object system creates the pipeline and the initialize_based_on_window call
    // regenerates it
    let mut object_system = ObjectSystem::new(
        queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
        [0.0, 0.0],
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    let mut ui_system = UiSystem::new(
        queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
        [0.0, 0.0],
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    object_system.register_object(m, glm::Mat4::identity());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = initialize_based_on_window(
        &memory_allocator,
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
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {e}"),
                        };

                    swapchain = new_swapchain;

                    let new_framebuffers = initialize_based_on_window(
                        &memory_allocator,
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
                        Err(AcquireError::OutOfDate) => {
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
                        SubpassContents::SecondaryCommandBuffers,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .execute_commands(object_system.draw(&cam))
                    .unwrap()
                    .execute_commands(ui_system.draw())
                    .unwrap();

                // additional passes would go here
                builder.end_render_pass().unwrap();

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
                    Err(FlushError::OutOfDate) => {
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
    memory_allocator: &StandardMemoryAllocator,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    object_system: &mut ObjectSystem<StandardMemoryAllocator>,
    ui_system: &mut UiSystem<StandardMemoryAllocator>,
    viewport: &mut Viewport,
    cam: &mut Camera,
) -> Vec<Arc<Framebuffer>> {
    let dimensions_u32 = images[0].dimensions().width_height();
    let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

    *viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions,
        depth_range: 0.0..1.0,
    };

    cam.aspect = dimensions[0] / dimensions[1];

    object_system.regenerate(dimensions);
    ui_system.regenerate(dimensions);

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(memory_allocator, dimensions_u32, Format::D16_UNORM).unwrap(),
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

pub fn depth_screenshots(m: MeshData, dim: (u32, u32), pos: &[Vec3]) -> Vec<String> {
    // I'm using the Vulkano examples to learn here
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/triangle.rs
    //
    // This example has some relevant information for offscreen rendering and saving
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/msaa-renderpass.rs
    //
    // 2023-07-07: Learning about how best to architect rendering is interesting. I've used these
    // pages to learn more. Not all of these were useful, but they were all interesting.
    //
    // https://vkguide.dev/docs/gpudriven/gpu_driven_engines/
    // https://advances.realtimerendering.com/s2020/RenderingDoomEternal.pdf
    // https://on-demand.gputechconf.com/gtc/2013/presentations/S3032-Advanced-Scenegraph-Rendering-Pipeline.pdf
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let (device, queue) = initialize_device(device_extensions);

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    const IMAGE_FORMAT: Format = Format::R32G32B32A32_SFLOAT;

    let image = StorageImage::new(
        &memory_allocator,
        vulkano::image::ImageDimensions::Dim2d {
            width: dim.0,
            height: dim.1,
            array_layers: 1,
        },
        IMAGE_FORMAT,
        Some(queue.queue_family_index()),
    )
    .unwrap();

    let transfer_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        (0..dim.0
            * dim.1
            * (IMAGE_FORMAT.block_size().unwrap() / std::mem::size_of::<f32>() as u64) as u32)
            .map(|_| 0f32),
    )
    .unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
            load: Clear,
            store: Store,
            format: IMAGE_FORMAT,
            samples: 1,
        },

        depth: {
            load: Clear,
            store: DontCare,
            format: vulkano::format::Format::D16_UNORM,
            samples: 1
        }
    },
        pass: {
            color: [color],

            depth_stencil: {depth},
        }
    )
    .unwrap();

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let object_subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    let mut object_system = ObjectSystem::new(
        queue.clone(),
        object_subpass,
        [dim.0 as f32, dim.1 as f32],
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    object_system.register_object(m, glm::identity());

    let framebuffer = {
        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(&memory_allocator, [dim.0, dim.1], Format::D16_UNORM)
                .unwrap(),
        )
        .unwrap();

        let view = ImageView::new_default(image.clone()).unwrap();
        Framebuffer::new(
            render_pass,
            render_pass::FramebufferCreateInfo {
                attachments: vec![view, depth_buffer],
                ..Default::default()
            },
        )
        .unwrap()
    };

    // initialization done!

    let screenshot_format = image::ImageFormat::OpenExr;
    let dir = screenshot_dir().unwrap();
    let mut ret = vec![];
    let mut cam = Camera::new(dim.0 as f32 / dim.1 as f32);
    for (img_num, pos) in pos.iter().enumerate() {
        cam.pos = *pos;

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
                    clear_values: vec![Some([1.0, 1.0, 1.0, 1.0].into()), Some(1.0.into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap();

        builder.execute_commands(object_system.draw(&cam)).unwrap();

        builder
            .end_render_pass()
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                transfer_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let buffer_content = transfer_buffer.read().unwrap();
        let file = format!(
            "{}/{}.{}",
            dir,
            img_num,
            screenshot_format.extensions_str()[0]
        );
        Rgba32FImage::from_raw(dim.0, dim.1, buffer_content.to_vec())
            .unwrap()
            .save_with_format(&file, screenshot_format)
            .unwrap();
        ret.push(file)
    }

    ret
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

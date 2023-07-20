use std::sync::Arc;

use image::Rgba32FImage;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};

use vulkano::device::{DeviceExtensions, DeviceOwned};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, StorageImage, SwapchainImage};

use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};

use vulkano::shader::ShaderModule;
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

use crate::mesh::{MeshData, MeshDataBuffs};

use crate::vkrender::init::initialize_device_window;
use crate::vkrender::render_systems::object_system::{ObjectSystem, ObjectSystemConfig, ObjectSystemRenderMode};
use crate::vkrender::{screenshot_dir, VkVertex};

pub fn depth_compare(m: MeshData, dim: (u32, u32), pos: &[[Vec3; 2]]) -> Vec<f32> {
    // I'm using the Vulkano examples to learn here
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/triangle.rs
    //
    // This example has some relevant information for offscreen rendering and saving
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/msaa-renderpass.rs

    let mut cam = Camera::new(dim.0 as f32 / dim.1 as f32);

    let device_extensions = DeviceExtensions {
        // for saving back to CPU memory
        khr_storage_buffer_storage_class: true,

        // for blending
        // I am not using this for now because of the lack of wrapper in Vulkano - there are some
        // extra invariants that using the extension requires.
        // see: https://github.com/vulkano-rs/vulkano/issues/572
        // see: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_blend_operation_advanced.html
        // ext_blend_operation_advanced: true,
        ..DeviceExtensions::empty()
    };

    let (device, queue) = crate::vkrender::init::initialize_device(device_extensions);

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image_format = Format::R32G32B32A32_SFLOAT;

    let image = StorageImage::new(
        &memory_allocator,
        vulkano::image::ImageDimensions::Dim2d {
            width: dim.0,
            height: dim.1,
            array_layers: 1,
        },
        image_format,
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
            * (image_format.block_size().unwrap() / std::mem::size_of::<f32>() as u64) as u32)
            .map(|_| 0f32),
    )
    .unwrap();

    let fs_cmp = cmp_fs::load(device.clone()).unwrap();
    let fs_cmp_entry = fs_cmp.entry_point("main").unwrap();
    let vs_cmp = cmp_vs::load(device.clone()).unwrap();
    let vs_cmp_entry = vs_cmp.entry_point("main").unwrap();

    // now to initialize the pipelines - this is completely foreign to opengl
    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
        final_color: {
            load: Clear,
            store: Store,
            format: image_format,
            samples: 1,
        },

        color_1: {
            load: Clear,
            store: DontCare,
            format: image_format,
            samples: 1,
        },

        color_2: {
            load: Clear,
            store: DontCare,
            format: image_format,
            samples: 1,
        },
        depth_1: {
            load: Clear,
            store: DontCare,
            format: vulkano::format::Format::D16_UNORM,
            samples: 1
        },
        depth_2: {
            load: Clear,
            store: DontCare,
            format: vulkano::format::Format::D16_UNORM,
            samples: 1
        }
    },
        passes: [
            {
                color: [color_1],
                depth_stencil: {depth_1},
                input: [],
            },
            {
                color: [color_2],
                depth_stencil: {depth_2},
                input: [],
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color_1, color_2],
            },
        ]
    )
    .unwrap();

    let depth_buffer_1 = ImageView::new_default(
        AttachmentImage::transient(&memory_allocator, [dim.0, dim.1], Format::D16_UNORM).unwrap(),
    )
    .unwrap();
    let depth_buffer_2 = ImageView::new_default(
        AttachmentImage::transient(&memory_allocator, [dim.0, dim.1], Format::D16_UNORM).unwrap(),
    )
    .unwrap();
    let color_buffer_1 = ImageView::new_default(
        AttachmentImage::with_usage(
            &memory_allocator,
            [dim.0, dim.1],
            image_format,
            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
        )
        .unwrap(),
    )
    .unwrap();
    let color_buffer_2 = ImageView::new_default(
        AttachmentImage::with_usage(
            &memory_allocator,
            [dim.0, dim.1],
            image_format,
            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
        )
        .unwrap(),
    )
    .unwrap();

    let (frame_vertex_buffer, frame_index_buffer) = {
        let frame_mesh: MeshDataBuffs<VkVertex> = crate::mesh::primative::frame().into();
        frame_mesh.to_buffers(&memory_allocator)
    };

    let pipeline_view = GraphicsPipeline::start()
        .render_pass(Subpass::from(render_pass.clone(), 2).unwrap())
        .vertex_input_state(VkVertex::per_vertex())
        .input_assembly_state(InputAssemblyState::new())
        .vertex_shader(vs_cmp_entry, ())
        .fragment_shader(fs_cmp_entry, ())
        .depth_stencil_state(DepthStencilState::disabled())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            Viewport {
                origin: [0.0, 0.0],
                dimensions: [dim.0 as f32, dim.1 as f32],
                depth_range: 0.0..1.0,
            },
        ]))
        .build(memory_allocator.device().clone())
        .unwrap();

    let framebuffer = {
        let view = ImageView::new_default(image.clone()).unwrap();
        Framebuffer::new(
            render_pass.clone(),
            render_pass::FramebufferCreateInfo {
                attachments: vec![
                    view,
                    color_buffer_1.clone(),
                    color_buffer_2.clone(),
                    depth_buffer_1,
                    depth_buffer_2,
                ],
                ..Default::default()
            },
        )
        .unwrap()
    };

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let dimensions = [dim.0 as f32, dim.1 as f32];
    // need 2 because they are on different subpasses
    let mut object_system_left = ObjectSystem::new_with_config(
        ObjectSystemConfig {
            render_mode: ObjectSystemRenderMode::Depth,
            ..Default::default()
        },
        queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
        dimensions,
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );
    let mut object_system_right = ObjectSystem::new_with_config(
        ObjectSystemConfig {
            render_mode: ObjectSystemRenderMode::Depth,
            ..Default::default()
        },
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
        dimensions,
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    {
        let instances = object_system_left.register_object(m, glm::Mat4::identity());
        object_system_right.register_object_instances(instances);
    }

    // initialization done!

    let screenshot_format = image::ImageFormat::OpenExr;
    let dir = screenshot_dir().unwrap();
    let mut ret = vec![];
    for (img_num, pos_pair) in pos.iter().enumerate() {
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let layout_view = pipeline_view.layout().set_layouts().get(0).unwrap();
        let set_view = {
            PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout_view.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer_1.clone()),
                    WriteDescriptorSet::image_view(1, color_buffer_2.clone()),
                ],
            )
            .unwrap()
        };

        // we are allowed to not do a render pass if we use dynamic
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 1.0].into()),
                        Some([1.0, 1.0, 1.0, 1.0].into()),
                        Some([1.0, 1.0, 1.0, 1.0].into()),
                        Some(1.0.into()),
                        Some(1.0.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap()
            .execute_commands(object_system_left.draw({
                cam.pos = pos_pair[0];
                &cam
            }))
            .unwrap()
            .next_subpass(SubpassContents::SecondaryCommandBuffers)
            .unwrap()
            .execute_commands(object_system_right.draw({
                cam.pos = pos_pair[1];
                &cam
            }))
            .unwrap()
            .next_subpass(SubpassContents::Inline)
            .unwrap()
            .bind_vertex_buffers(0, frame_vertex_buffer.clone())
            .bind_index_buffer(frame_index_buffer.clone())
            .bind_pipeline_graphics(pipeline_view.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline_view.layout().clone(),
                0,
                set_view,
            )
            .draw_indexed(frame_index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap_or_else(|e| panic!("{}", e))
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
        // todo: change to get pixel root mse
        /*
        Rgba32FImage::from_raw(dim.0, dim.1, buffer_content.to_vec())
            .unwrap()
            .save_with_format(&file, screenshot_format)
            .unwrap();
        ret.push(file)
        */

        let prmse = buffer_content.iter().filter(|p| p > &&0.0).sum::<f32>() / 
        buffer_content.iter().filter(|p| p > &&0.0).count() as f32;

        ret.push(prmse);
    }

    ret
}

const INTERMEDIATE_IMAGE_FORMAT: Format = vulkano::format::Format::R32G32B32A32_SFLOAT;
pub fn display_duel_render(m: MeshData, orbit_amt: glm::Vec2) {
    let (device, queue, surface, event_loop) = initialize_device_window(DeviceExtensions {
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

    let fs_cmp = cmp_fs::load(device.clone()).unwrap();
    let vs_cmp = cmp_vs::load(device.clone()).unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let (frame_vertex_buffer, frame_index_buffer) = {
        let frame_mesh: MeshDataBuffs<VkVertex> = crate::mesh::primative::frame().into();
        frame_mesh.to_buffers(&memory_allocator)
    };

    let mut cam = Camera::new(1.0);
    

    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
        final_color: {
            load: Clear,
            store: Store,
            format: swapchain.image_format(),
            samples: 1,
        },

        color_1: {
            load: Clear,
            store: DontCare,
            format: INTERMEDIATE_IMAGE_FORMAT,
            samples: 1,
        },

        color_2: {
            load: Clear,
            store: DontCare,
            format: INTERMEDIATE_IMAGE_FORMAT,
            samples: 1,
        },
        depth_1: {
            load: Clear,
            store: DontCare,
            format: vulkano::format::Format::D16_UNORM,
            samples: 1
        },
        depth_2: {
            load: Clear,
            store: DontCare,
            format: vulkano::format::Format::D16_UNORM,
            samples: 1
        }
    },
        passes: [
            {
                color: [color_1],
                depth_stencil: {depth_1},
                input: [],
            },
            {
                color: [color_2],
                depth_stencil: {depth_2},
                input: [],
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color_1, color_2],
            },
        ]
    )
    .unwrap();

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // TODO: creating an object system creates the pipeline and the initialize_based_on_window call
    // regenerates it
    let mut object_system_left = ObjectSystem::new_with_config(
        ObjectSystemConfig {
            render_mode: ObjectSystemRenderMode::Depth,
            ..Default::default()
        },
        queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
        [0.0, 0.0],
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    let mut object_system_right = ObjectSystem::new_with_config(
        ObjectSystemConfig {
            render_mode: ObjectSystemRenderMode::Depth,
            ..Default::default()
        },
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
        [0.0, 0.0],
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    {
        let instances = object_system_left.register_object(m, glm::Mat4::identity());
        object_system_right.register_object_instances(instances);
    }

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let (mut set_view, mut pipeline_view, mut framebuffers) = initialize_based_on_window(
        &memory_allocator,
        &descriptor_set_allocator,
        &images,
        render_pass.clone(),
        &mut [&mut object_system_left, &mut object_system_right],
        &mut viewport,
        &mut cam,
        (fs_cmp.clone(), vs_cmp.clone()),
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
            Event::RedrawEventsCleared => {
                // do not draw if size is zero (eg minimized)
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
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

                    (set_view, pipeline_view, framebuffers) = initialize_based_on_window(
                        &memory_allocator,
                        &descriptor_set_allocator,
                        &new_images,
                        render_pass.clone(),
                        &mut [&mut object_system_left, &mut object_system_right],
                        &mut viewport,
                        &mut cam,
                        (fs_cmp.clone(), vs_cmp.clone()),
                    );
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

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // we are allowed to not do a render pass if we use dynamic

                let mut temp_cam = cam.clone();
                temp_cam.orbit_target(&orbit_amt);
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some([1.0, 1.0, 1.0, 1.0].into()),
                                Some([1.0, 1.0, 1.0, 1.0].into()),
                                Some(1.0.into()),
                                Some(1.0.into()),
                            ],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassContents::SecondaryCommandBuffers,
                    )
                    .unwrap()
                    .execute_commands(object_system_left.draw(&cam))
                    .unwrap()
                    .next_subpass(SubpassContents::SecondaryCommandBuffers)
                    .unwrap()
                    .execute_commands(object_system_right.draw(&temp_cam))
                    .unwrap()
                    .next_subpass(SubpassContents::Inline)
                    .unwrap()
                    .bind_vertex_buffers(0, frame_vertex_buffer.clone())
                    .bind_index_buffer(frame_index_buffer.clone())
                    .bind_pipeline_graphics(pipeline_view.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline_view.layout().clone(),
                        0,
                        set_view.clone(),
                    )
                    .draw_indexed(frame_index_buffer.len() as u32, 1, 0, 0, 0)
                    .unwrap_or_else(|e| panic!("{}", e))
                    .end_render_pass()
                    .unwrap();

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
    descset_allocator: &StandardDescriptorSetAllocator,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    object_systems: &mut [&mut ObjectSystem<StandardMemoryAllocator>],
    viewport: &mut Viewport,
    cam: &mut Camera,
    (fs, vs): (Arc<ShaderModule>, Arc<ShaderModule>),
) -> (
    Arc<PersistentDescriptorSet>,
    Arc<GraphicsPipeline>,
    Vec<Arc<Framebuffer>>,
) {
    let dimensions_u32 = images[0].dimensions().width_height();
    let dim = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

    cam.aspect = dim[0] / dim[1];

    *viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: dim,
        depth_range: 0.0..1.0,
    };

    for object_system in object_systems {
        object_system.regenerate(dim);
    }

    let depth_buffer_1 = ImageView::new_default(
        AttachmentImage::transient(memory_allocator, dimensions_u32, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let depth_buffer_2 = ImageView::new_default(
        AttachmentImage::transient(memory_allocator, dimensions_u32, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let color_buffer_1 = ImageView::new_default(
        AttachmentImage::with_usage(
            memory_allocator,
            dimensions_u32,
            INTERMEDIATE_IMAGE_FORMAT,
            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
        )
        .unwrap(),
    )
    .unwrap();

    let color_buffer_2 = ImageView::new_default(
        AttachmentImage::with_usage(
            memory_allocator,
            dimensions_u32,
            INTERMEDIATE_IMAGE_FORMAT,
            ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::INPUT_ATTACHMENT,
        )
        .unwrap(),
    )
    .unwrap();

    let pipeline_view = GraphicsPipeline::start()
        .render_pass(Subpass::from(render_pass.clone(), 2).unwrap())
        .vertex_input_state(VkVertex::per_vertex())
        .input_assembly_state(InputAssemblyState::new())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::disabled())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            Viewport {
                origin: [0.0, 0.0],
                dimensions: dim,
                depth_range: 0.0..1.0,
            },
        ]))
        .build(memory_allocator.device().clone())
        .unwrap();

    let layout_view = pipeline_view.layout().set_layouts().get(0).unwrap();
    let set_view = {
        PersistentDescriptorSet::new(
            descset_allocator,
            layout_view.clone(),
            [
                WriteDescriptorSet::image_view(0, color_buffer_1.clone()),
                WriteDescriptorSet::image_view(1, color_buffer_2.clone()),
            ],
        )
        .unwrap()
    };

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                render_pass::FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        color_buffer_1.clone(),
                        color_buffer_2.clone(),
                        depth_buffer_1.clone(),
                        depth_buffer_2.clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    (set_view, pipeline_view, framebuffers)
}

mod cmp_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/compare_vk.glsl"
    }
}

mod cmp_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/compare_vs.glsl"
    }
}

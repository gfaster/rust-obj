use std::sync::Arc;

use vulkano::buffer::{BufferContents, Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{DeviceExtensions, QueueFlags, Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage, ImageAccess};
use vulkano::instance::Instance;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::{VulkanLibrary, render_pass, sync};
use vulkano::memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryUsage};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::{self, Vertex};
use vulkano::pipeline::graphics::viewport::{ViewportState, Viewport};
use vulkano::render_pass::{Subpass, Framebuffer, RenderPass};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo, SwapchainCreationError, acquire_next_image, AcquireError, SwapchainPresentInfo};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};

use crate::glm::Vec3;
use crate::mesh::{self, MeshData, MeshDataBuffs};
use crate::mesh::mtl::Material;



// format attributes use [`vulkano::format::Format`] enum fields 
#[derive(BufferContents, vertex_input::Vertex)]
#[repr(C)]
struct VkVertex {
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
            tex: value.tex.into()
        }
    }
}


pub fn display_model(m: mesh::MeshData) {
    // I'm using the Vulkano examples to learn here
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/triangle.rs
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);

    let instance = Instance::new(
        library,
        vulkano::instance::InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    let event_loop = EventLoop::new();

    // instance maybe needs to be cloned?
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // find a physical device that supports all the features we need
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no supported physical device found");

    eprintln!(
        "Using device {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    // initialization of device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    ).unwrap();

    // only have one queue, so we just use that
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilites = device.physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        
        let image_format = Some(device.physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0].0);

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

                composite_alpha: surface_capabilites.supported_composite_alpha.into_iter().next().unwrap(),

                ..Default::default()
            }
        )
        .unwrap()
    };

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // setup done - now to mesh

    let MeshDataBuffs {verts: vertices, indices}: MeshDataBuffs<VkVertex> = m.into(); 

    let vertex_buffer = Buffer::from_iter(
        &memory_allocator, 
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        vertices
        ).unwrap();
    let index_buffer = Buffer::from_iter(
        &memory_allocator, 
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        indices
        ).unwrap();

    // supposedly we could do shaders at compile time - but there is a whole mess 
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "shaders/vert_vk.glsl",
            linalg_type: "nalgebra"
        }
    }
    mod fs {
        use super::Material;

        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/frag_vk.glsl",
            linalg_type: "nalgebra"
        }

        // declaration of ShaderMtl from macro parsing of frag_vk.glsl
        impl From<Material> for ShaderMtl {
            fn from(value: Material) -> Self {
                ShaderMtl {
                    base_diffuse: value.diffuse().into(),
                    base_ambient: value.ambient().into(),
                    base_specular: value.specular().into(),
                    base_specular_factor: value.base_specular_factor().into(),
                }
            }
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // now to initialize the pipelines - this is completely foreign to opengl
    let render_pass = vulkano::single_pass_renderpass!(
    device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
        },
        pass: {
            color: [color],

            // no depth/stencil attachment for now
            depth_stencil: {},
        }
    ).unwrap();

    let pipeline = GraphicsPipeline::start()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .vertex_input_state(VkVertex::per_vertex())
        // list of triangles - I think this doesn't need to change for idx buf
        .input_assembly_state(InputAssemblyState::new())
        // entry point isn't necessarily main
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        // resizable window
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .build(device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = initialize_based_on_window(&images, render_pass.clone(), &mut viewport);

    let command_buffer_allocator = 
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    // initialization done!

    // swapchain can be spuriously invalidated (window resized)
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow|{
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { *control_flow = ControlFlow::Exit },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => { recreate_swapchain = true },
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
                    let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                        image_extent: dimensions.into(),
                        ..swapchain.create_info()
                    }) {
                            Ok(r) => r,
                                // can happen when resizing -> easy way to fix is just restart loop
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {e}"),
                        };

                    swapchain = new_swapchain;

                    framebuffers = initialize_based_on_window(&new_images, render_pass.clone(), &mut viewport);

                    recreate_swapchain = false;
                }

                // need to acquire image before drawing, blocks if it's not ready yet (too many
                // commands), so it has optional timeout
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(swapchain.clone(), None) {
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

                let mut builder = AutoCommandBufferBuilder::primary(&command_buffer_allocator, queue.queue_family_index(), CommandBufferUsage::OneTimeSubmit).unwrap();

                // we can not do a render pass if we use dynamic
                builder.begin_render_pass(RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffers[image_index as usize].clone())
                }, SubpassContents::Inline).unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_index_buffer(index_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    // additional passes would go here
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index))
                    .then_signal_fence_and_flush();

                match future {
                    Ok(f) => {
                        previous_frame_end = Some(f.boxed());
                    },
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
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images.iter()
    .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                render_pass::FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                }
            ).unwrap()
        }).collect::<Vec<_>>()
}

pub fn depth_screenshots(_m: MeshData, _dim: (u32, u32), _pos: &[Vec3]) -> Vec<String> {
    todo!()
}

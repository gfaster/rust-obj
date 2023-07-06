use std::sync::Arc;

use image::Rgba32FImage;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, StorageImage};
use vulkano::instance::Instance;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, Subpass};


use vulkano::sync::GpuFuture;
use vulkano::{render_pass, sync, VulkanLibrary};

use crate::controls::Camera;
use crate::glm::Vec3;

use crate::mesh::{MeshData, MeshDataBuffs};

use crate::vkrender::{fs, vs, screenshot_dir, VkVertex, generate_uniforms};

pub fn depth_compare(m: MeshData, dim: (u32, u32), pos: &[[Vec3; 2]]) -> Vec<String> {
    // I'm using the Vulkano examples to learn here
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/triangle.rs
    //
    // This example has some relevant information for offscreen rendering and saving
    // https://github.com/vulkano-rs/vulkano/blob/0.33.X/examples/src/bin/msaa-renderpass.rs
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

    let mut cam = Camera::new();

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
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
                .position(|(_, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
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
        },
    )
    .unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // only have one queue, so we just use that
    let queue = queues.next().unwrap();

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

    // setup done - now to mesh
    let mesh_meta = m.get_meta();
    let MeshDataBuffs {
        verts: vertices,
        indices,
    }: MeshDataBuffs<VkVertex> = m.into();

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
        vertices,
    )
    .unwrap();
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
        indices,
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

    let uniform_buffer = SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
    );

    let vs = vs::load(device.clone()).unwrap();
    let vs_entry = vs.entry_point("main").unwrap();
    let fs = fs::load(device.clone()).unwrap();
    let fs_entry = fs.entry_point("main").unwrap();

    // now to initialize the pipelines - this is completely foreign to opengl
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
            load: Clear,
            store: Store,
            format: image_format,
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

    let (pipeline, framebuffer) = {
        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(&memory_allocator, [dim.0, dim.1], Format::D16_UNORM)
                .unwrap(),
        )
        .unwrap();

        let pipeline = GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_state(VkVertex::per_vertex())
            // list of triangles - I think this doesn't need to change for idx buf
            .input_assembly_state(InputAssemblyState::new())
            // entry point isn't necessarily main
            .vertex_shader(vs_entry, ())
            .fragment_shader(fs_entry, ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
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
                render_pass,
                render_pass::FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer],
                    ..Default::default()
                },
            )
            .unwrap()
        };
        (pipeline, framebuffer)
    };

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    // initialization done!

    let screenshot_format = image::ImageFormat::OpenExr;
    let dir = screenshot_dir().unwrap();
    let mut ret = vec![];
    for (img_num, pos_pair) in pos.iter().enumerate() {
        cam.pos = pos_pair[0];

        let aspect = dim.0 as f32 / dim.1 as f32;
        let (s_cam, s_mat, s_mtl, s_light) = generate_uniforms(&mesh_meta, &cam, aspect);
        let layout = pipeline.layout().set_layouts().get(0).unwrap();

        macro_rules! gen_write_descriptors {
        ($uniform:ident, $(($data:expr, $binding:expr)),*) => {
            [
                $( {
                    let subbuffer = $uniform.allocate_sized().unwrap();
                    *subbuffer.write().unwrap() = $data;
                    WriteDescriptorSet::buffer(
                        $binding, subbuffer)
                }, )*
            ]
            }
        }

        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout.clone(),
            gen_write_descriptors!(
                uniform_buffer,
                (s_cam, vs::CAM_BINDING),
                (s_mat, vs::MAT_BINDING),
                (s_mtl, fs::MTL_BINDING),
                (s_light, fs::LIGHT_BINDING)
            ),
        )
        .unwrap();

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
                SubpassContents::Inline,
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set,
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .bind_index_buffer(index_buffer.clone())
            .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap()
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

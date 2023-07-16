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
use vulkano::image::{AttachmentImage, ImageUsage, StorageImage};

use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, Subpass};


use vulkano::sync::GpuFuture;
use vulkano::{render_pass, sync};

use crate::controls::Camera;
use crate::glm::Vec3;

use crate::mesh::{MeshData, MeshDataBuffs};

use crate::vkrender::render_systems::object_system::ObjectSystem;
use crate::vkrender::{ screenshot_dir,  VkVertex};

pub fn depth_compare(m: MeshData, dim: (u32, u32), pos: &[[Vec3; 2]]) -> Vec<String> {
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
    let command_buffer_allocator =
        Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

    let dimensions = [dim.0 as f32, dim.1 as f32];
    // need 2 because they are on different subpasses
    let mut object_system_left = ObjectSystem::new(queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
        dimensions,
        memory_allocator.clone(),
        command_buffer_allocator.clone(), 
        descriptor_set_allocator.clone()
    );
    let mut object_system_right = ObjectSystem::new(queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
        dimensions,
        memory_allocator.clone(),
        command_buffer_allocator.clone(), 
        descriptor_set_allocator.clone()
    );

    {
        let instance = object_system_left.register_object(m, glm::Mat4::identity());
        object_system_right.register_object_instance(instance);
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
        Rgba32FImage::from_raw(dim.0, dim.1, buffer_content.to_vec())
            .unwrap()
            .save_with_format(&file, screenshot_format)
            .unwrap();
        ret.push(file)
    }

    ret
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

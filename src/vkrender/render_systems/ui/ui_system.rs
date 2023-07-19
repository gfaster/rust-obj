//! I think this is mostly superflous, but I'm basically doing this because it's a neat exercise

use std::cell::Cell;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
    PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, SecondaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageDimensions, ImmutableImage};
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::{input_assembly::InputAssemblyState, viewport::Viewport};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::PipelineBindPoint;

use vulkano::sampler::{Sampler, SamplerCreateInfo};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    memory::allocator::MemoryAllocator,
    pipeline::{graphics::viewport::ViewportState, GraphicsPipeline},
    render_pass::Subpass,
};

use crate::mesh::{MeshData, MeshDataBuffs};
use crate::util::CellMap;
use crate::{
    controls::Camera,
    mesh::{mtl::Material, MeshMeta},
};

use crate::vkrender::VkVertex;


/// Basic text rendering
///
/// ## Usage
/// - create a new `UiSystem`
/// - set text
/// - draw
///
/// Goal: https://vkguide.dev/docs/gpudriven/gpu_driven_engines/
/// Stretch goal: https://www.nvidia.com/content/GTC-2010/pdfs/2152_GTC2010.pdf
pub struct UiSystem<Allocator>
where
    Arc<Allocator>: MemoryAllocator,
{
    text: Box<[u32; 2400]>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,

    /// command buffer for textures that need to be uploaded to the GPU. uploading happens on draw,
    /// if there are any pending
    upload_cmdbuf: Cell<
        Option<
            AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>,
        >,
    >,

    /// various necessary vulkano components
    gfx_queue: Arc<Queue>,
    cmdbuf_allocator: Arc<StandardCommandBufferAllocator>,
    descset_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<Allocator>,

    /// the subbuffer allocator is owned because I think it may be better for reducing
    /// fragmentation
    uniform_buffer: SubbufferAllocator<Arc<Allocator>>,
    vertex_buffer: Subbuffer<[VkVertex]>
}

macro_rules! subbuffer_write_descriptors {
    ($uniform:expr, $(($data:expr, $binding:expr)),*) => {
        [ $( {
            let subbuffer = $uniform.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = $data;
            WriteDescriptorSet::buffer(
                $binding, subbuffer)
        }, )* ]
    }
}

impl<Allocator> UiSystem<Allocator>
where
    Arc<Allocator>: MemoryAllocator
{
    
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        dimensions: [f32; 2],
        memory_allocator: Arc<Allocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        let vs = vs::load(gfx_queue.device().clone()).unwrap();
        let fs = fs::load(gfx_queue.device().clone()).unwrap();

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let frame_mesh = Into::<MeshDataBuffs<VkVertex>>::into(crate::mesh::primative::frame());
        let (vertex_buffer, _) = frame_mesh.to_buffers(&memory_allocator);

        let pipeline = GraphicsPipeline::start()
            .render_pass(subpass.clone())
            .vertex_input_state(VkVertex::per_vertex())
            .input_assembly_state(InputAssemblyState::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions,
                    depth_range: 0.0..1.0,
                },
            ]))
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .build(gfx_queue.device().clone())
            .unwrap();

        Self {
            text: Box::new([0; 2400]),
            pipeline,
            subpass,
            cmdbuf_allocator: command_buffer_allocator,
            descset_allocator: descriptor_set_allocator,
            memory_allocator,
            gfx_queue,
            uniform_buffer,
            vertex_buffer,
            upload_cmdbuf: None.into(),
        }
    }

    pub fn draw(&self, cam: &Camera) -> SecondaryAutoCommandBuffer {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap().clone();
        let set = PersistentDescriptorSet::new(
            &self.descset_allocator,
            layout,
            subbuffer_write_descriptors! {
                self.uniform_buffer,
            },
        )
        .unwrap();

        self.upload_cmdbuf.take().map(|c| {
            log!("uploading font");
            c.build().unwrap().execute(self.gfx_queue.clone())
        });

        let sampler = Sampler::new(
            self.gfx_queue.device().clone(),
            SamplerCreateInfo::simple_repeat_linear(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::secondary(
            &self.cmdbuf_allocator,
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(6, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    pub fn regenerate(&mut self, dimensions: [f32; 2]) {
        log!("regenerating...");
        let vs = vs::load(self.gfx_queue.device().clone()).unwrap();
        let fs = fs::load(self.gfx_queue.device().clone()).unwrap();

        let pipeline = GraphicsPipeline::start()
            .render_pass(self.subpass.clone())
            .vertex_input_state(VkVertex::per_vertex())
            .input_assembly_state(InputAssemblyState::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions,
                    depth_range: 0.0..1.0,
                },
            ]))
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .build(self.gfx_queue.device().clone())
            .unwrap();

        self.pipeline = pipeline;
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/compare_vs.glsl",
        linalg_type: "nalgebra",
        custom_derives: [Debug]
    }

    // TODO: this might be revealed after entry point is added
    pub const MAT_BINDING: u32 = 0;
    pub const CAM_BINDING: u32 = 1;
}
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/text_vk.glsl",
        linalg_type: "nalgebra",
        custom_derives: [Debug]
    }

    // TODO: this might be revealed after entry point is added
    pub const FONT_BINDING: u32 = 2;
    pub const TEXT_BINDING: u32 = 3;
    pub const SCREEN_BINDING: u32 = 4;
}

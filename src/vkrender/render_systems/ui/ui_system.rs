//! I think this is mostly superflous, but I'm basically doing this because it's a neat exercise

use std::io::{BufReader, Write};
use std::mem::size_of;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
    SecondaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};

use vulkano::device::Device;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::{input_assembly::InputAssemblyState, viewport::Viewport};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::{Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};

use vulkano::shader::EntryPoint;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    pipeline::{graphics::viewport::ViewportState, GraphicsPipeline},
    render_pass::Subpass,
};

use crate::vkrender::VkVertex;

const TEXT_CELLS: usize = 2400;

/// Basic text rendering
///
/// ## Usage
/// - create a new `UiSystem`
/// - set text
/// - draw
///
/// Goal: https://vkguide.dev/docs/gpudriven/gpu_driven_engines/
/// Stretch goal: https://www.nvidia.com/content/GTC-2010/pdfs/2152_GTC2010.pdf
pub struct UiSystem {
    text: Box<[u32; TEXT_CELLS]>,
    cursor: usize,
    font_meta: fs::ShaderFontMeta,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,

    /// various necessary vulkano components
    gfx_queue: Arc<Queue>,
    cmdbuf_allocator: Arc<StandardCommandBufferAllocator>,
    descset_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,

    /// the subbuffer allocator is owned because I think it may be better for reducing
    /// fragmentation
    uniform_buffer: SubbufferAllocator<StandardMemoryAllocator>,
    vertex_buffer: Subbuffer<[VkVertex]>,
    bitmap_buffer: Subbuffer<[u32]>,
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

impl UiSystem {
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        dimensions: [f32; 2],
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        let vs = vs::load(gfx_queue.device().clone()).unwrap();
        let fs = fs::load(gfx_queue.device().clone()).unwrap();

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        );

        let bdf = include_str!("../../../../assets/ter-u12b.bdf");
        let mut reader = BufReader::new(bdf.as_bytes());
        let font = super::font::Font::parse_bdf(&mut reader).unwrap_or_else(|e| panic!("{}", e));

        let bitmap_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            font.flat_bitmaps(),
        )
        .unwrap();

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            crate::mesh::primitive::frame().to_tri_list(),
        )
        .unwrap();

        let pipeline = Self::make_pipeline(
            gfx_queue.device().clone(),
            subpass.clone(),
            vs.entry_point("main").unwrap(),
            fs.entry_point("main").unwrap(),
            dimensions,
        );

        Self {
            text: Box::new([0 as u32; 2400]),
            cursor: 0,
            pipeline,
            subpass,
            cmdbuf_allocator: command_buffer_allocator,
            descset_allocator: descriptor_set_allocator,
            memory_allocator,
            gfx_queue,
            uniform_buffer,
            vertex_buffer,
            bitmap_buffer,
            font_meta: font.into(),
        }
    }

    fn make_pipeline(
        device: Arc<Device>,
        subpass: Subpass,
        vs_entry: EntryPoint,
        fs_entry: EntryPoint,
        extent: [f32; 2],
    ) -> Arc<GraphicsPipeline> {
        let vertex_input_state = [VkVertex::per_vertex()]
            .definition(&vs_entry.info().input_interface)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs_entry.clone()),
            PipelineShaderStageCreateInfo::new(fs_entry.clone()),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        GraphicsPipeline::new(
            device,
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0, 0.0],
                        extent: [extent[0] as f32, extent[1] as f32],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                depth_stencil_state: Some(DepthStencilState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }

    pub fn draw(&self) -> Arc<SecondaryAutoCommandBuffer> {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap().clone();
        let set = PersistentDescriptorSet::new(
            &self.descset_allocator,
            layout,
            subbuffer_write_descriptors! {
                self.uniform_buffer,
                (self.font_meta.clone(), fs::FONT_META_BINDING),
                (*self.text_as_shader(), fs::TEXT_BINDING)
            }
            .into_iter()
            .chain([
                {
                    let subbuffer = self
                        .uniform_buffer
                        .allocate_slice(self.text.len() as vulkano::DeviceSize)
                        .unwrap();
                    subbuffer
                        .write()
                        .unwrap()
                        .copy_from_slice(self.text.as_slice());
                    WriteDescriptorSet::buffer(fs::TEXT_BINDING, subbuffer)
                },
                WriteDescriptorSet::buffer(fs::FONT_DATA_BINDING, self.bitmap_buffer.clone()),
            ]),
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::secondary(
            &*self.cmdbuf_allocator,
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
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
            .draw(6, 1, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    pub fn regenerate(&mut self, dimensions: [f32; 2]) {
        log!("regenerating...");
        let device = self.gfx_queue.device();
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let pipeline =
            Self::make_pipeline(device.clone(), self.subpass.clone(), vs, fs, dimensions);
        self.pipeline = pipeline;
    }

    fn text_as_shader(&self) -> Box<fs::ShaderText> {
        assert_eq!(size_of::<fs::ShaderText>(), size_of::<[u32; TEXT_CELLS]>());
        let chars;

        unsafe {
            chars = std::slice::from_raw_parts(self.text.as_ptr() as *const _, TEXT_CELLS / 4)
                .try_into()
                .unwrap();
        }

        Box::new(fs::ShaderText { chars })
    }

    pub fn clear(&mut self) {
        self.cursor = 0;
        self.text.fill(0);
    }

    pub fn contents(&self) -> String {
        self.text
            .chunks(80)
            .flat_map(|l| {
                let end = l.into_iter().position(|c| *c == 0).unwrap_or(80);
                l[..end]
                    .iter()
                    .map(|b| char::try_from(*b).unwrap_or(char::REPLACEMENT_CHARACTER))
                    .chain(['\n'])
            })
            .collect()
    }
}

impl Write for UiSystem {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut len = 0;

        for byte in buf {
            if self.cursor >= TEXT_CELLS {
                break;
            }
            let x = self.cursor % 80;

            if byte.is_ascii() && !byte.is_ascii_control() {
                self.text[self.cursor] = *byte as u32;
                self.cursor += 1;
            } else if *byte == b'\n' {
                self.cursor += 80 - x;
            } else {
                self.text[self.cursor] = 0;
                self.cursor += 1;
            }
            len += 1;
        }

        Ok(len)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/compare_vs.glsl",
        linalg_type: "nalgebra",
        custom_derives: [Debug]
    }
}
pub mod fs {
    use crate::vkrender::render_systems::ui::font::Font;

    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/text_vk.glsl",
        linalg_type: "nalgebra",
        custom_derives: [Debug, Clone]
    }

    impl From<Font> for ShaderFontMeta {
        fn from(value: Font) -> Self {
            Self {
                bbx_x: value.bbx[0],
                bbx_y: value.bbx[1],
                bbx_off_x: value.bbx[2],
                bbx_off_y: value.bbx[3],
                dwidth_x: value.bbx[0],
                dwidth_y: value.bbx[1],
                pixel_width: value.bbx[0],
                pixel_height: value.bbx[1],
                bytes_per_row: 1,
                bytes_per_glyph: value.bbx[1],
                size: 12.0.into(),
            }
        }
    }

    pub const FONT_META_BINDING: u32 = 0;
    pub const SCREEN_BINDING: u32 = 1;
    pub const TEXT_BINDING: u32 = 2;
    pub const FONT_DATA_BINDING: u32 = 3;
}

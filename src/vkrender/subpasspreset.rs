//! To render an object, we need to do several things:
//! - create a pipeline that has shaders, subpass, and descriptor sets
//! - create a renderpass that contains the subpass as well as attachments
//! - have subpass inputs and outputs bound
//! - potentially have samplers bound
//! - put it all together in a command buffer
//!
//! I'm making separate structs for this instead of a trait because of how different they are

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
use crate::{
    controls::Camera,
    mesh::{mtl::Material, MeshMeta},
};

use super::VkVertex;

/// Basic rendering of an object
///
/// ## Usage
/// - create a new `ObjectSystem`
/// - register meshes
/// - draw
///
/// Goal: https://vkguide.dev/docs/gpudriven/gpu_driven_engines/
/// Stretch goal: https://www.nvidia.com/content/GTC-2010/pdfs/2152_GTC2010.pdf
pub struct ObjectSystem<Allocator>
where
    Arc<Allocator>: MemoryAllocator,
{
    objects: Vec<Option<ObjectInstance>>,
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
    uniform_allocator: SubbufferAllocator<Arc<Allocator>>,
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

impl<Allocator> ObjectSystem<Allocator>
where
    Arc<Allocator>: MemoryAllocator,
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

        let subbuffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

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
            objects: Vec::new(),
            pipeline,
            subpass,
            cmdbuf_allocator: command_buffer_allocator,
            descset_allocator: descriptor_set_allocator,
            memory_allocator,
            gfx_queue,
            uniform_allocator: subbuffer_allocator,
            upload_cmdbuf: None.into(),
        }
    }

    pub fn draw(&self, cam: &Camera) -> SecondaryAutoCommandBuffer {
        let layout_0 = self.pipeline.layout().set_layouts().get(0).unwrap().clone();
        let layout_1 = self.pipeline.layout().set_layouts().get(1).unwrap().clone();
        let (s_light, s_cam) = generate_uniforms_0(cam);
        let set_0 = PersistentDescriptorSet::new(
            &self.descset_allocator,
            layout_0,
            subbuffer_write_descriptors! {
                self.uniform_allocator,
                (s_cam, vs::CAM_BINDING),
                (s_light, fs::LIGHT_BINDING)
            },
        )
        .unwrap();

        self.upload_cmdbuf.take().map(|c| {
            eprintln!("[ObjectSystem]: uploading textures");
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
                set_0,
            );

        for object in self.objects.iter().filter_map(|x| x.as_ref()) {
            let (s_mat, s_mtl) = generate_uniforms_1(&object.meta);
            let set_1 = PersistentDescriptorSet::new(
                &self.descset_allocator,
                layout_1.clone(),
                subbuffer_write_descriptors! {
                    self.uniform_allocator,
                    (s_mat, vs::MAT_BINDING),
                    (s_mtl, fs::MTL_BINDING)
                }
                .into_iter()
                .chain([WriteDescriptorSet::image_view_sampler(
                    fs::SAMPLER_BINDING,
                    object.texture.clone(),
                    sampler.clone(),
                )]),
            )
            .unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    1,
                    set_1,
                )
                .bind_vertex_buffers(0, object.vertices.clone())
                .bind_index_buffer(object.indices.clone())
                .draw_indexed(object.indices.len() as u32, 1, 0, 0, 0)
                .unwrap();
        }
        builder.build().unwrap()
    }

    pub fn regenerate(&mut self, dimensions: [f32; 2]) {
        log!("regenerated");
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

    pub fn register_object(&mut self, object: MeshData, transform: glm::Mat4) {
        let meta = object.get_meta();
        let mesh_buffs: MeshDataBuffs<VkVertex> = object.into();
        let (vertices, indices) = mesh_buffs.to_buffers(&self.memory_allocator);

        let texture = {
            // take because lifetimes are annoying here
            let mut upload = self.upload_cmdbuf.take().unwrap_or_else(|| {
                AutoCommandBufferBuilder::primary(
                    &self.cmdbuf_allocator,
                    self.gfx_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap()
            });

            let img = meta
                .material
                .diffuse_map()
                .unwrap_or_else(|| Material::dev_texture().into());
            let dimensions = ImageDimensions::Dim2d {
                width: img.width(),
                height: img.height(),
                array_layers: 1,
            };
            let image = ImmutableImage::from_iter(
                &self.memory_allocator,
                img.as_raw().clone(),
                dimensions,
                vulkano::image::MipmapsCount::Log2,
                Format::R8G8B8A8_SRGB,
                &mut upload,
            )
            .unwrap();

            self.upload_cmdbuf = Some(upload).into();

            ImageView::new_default(image).unwrap()
        };

        self.objects.push(Some(ObjectInstance {
            vertices,
            indices,
            meta,
            transform,
            texture,
        }));
    }
}

struct ObjectInstance {
    vertices: Subbuffer<[VkVertex]>,
    indices: Subbuffer<[u32]>,
    meta: MeshMeta,
    texture: Arc<ImageView<ImmutableImage>>,
    transform: glm::Mat4,
}

fn generate_uniforms_0(cam: &Camera) -> (fs::ShaderLight, vs::ShaderCamAttr) {
    let near = 1.0;
    let far = 4.0;
    let projection_matrix = glm::perspective(cam.aspect, 1.0, near, far);

    let cam_attr = vs::ShaderCamAttr {
        near,
        far,
        projection_matrix,
        transform: cam.get_transform(),
    };

    let light = fs::ShaderLight { light_pos: cam.pos };

    (light, cam_attr)
}

pub fn generate_uniforms_1(meta: &MeshMeta) -> (vs::ShaderMatBuffer, fs::ShaderMtl) {
    let scale = 1.5 / meta.normalize_factor;
    let center = meta.centroid;
    let transform = glm::Mat4::from([
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, scale, 0.0],
        [0.0, 0.0, 0.0, 1.0f32],
    ]) * glm::Mat4::from([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [center.x, center.y, center.z, 1.0],
    ]);

    let normal_matrix = glm::transpose(&glm::inverse(&transform));

    let mat = vs::ShaderMatBuffer {
        transform,
        normal_matrix,
    };

    let mtl = meta.material.clone().into();

    (mat, mtl)
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_vk.glsl",
        linalg_type: "nalgebra",
        custom_derives: [Debug]
    }

    // TODO: this might be revealed after entry point is added
    pub const MAT_BINDING: u32 = 0;
    pub const CAM_BINDING: u32 = 1;
}
pub mod fs {
    use super::Material;

    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag_vk.glsl",
        linalg_type: "nalgebra",
        custom_derives: [Debug]
    }

    // TODO: this might be revealed after entry point is added
    pub const MTL_BINDING: u32 = 2;
    pub const LIGHT_BINDING: u32 = 3;
    pub const SAMPLER_BINDING: u32 = 4;

    // declaration of ShaderMtl from macro parsing of frag_vk.glsl
    impl From<Material> for ShaderMtl {
        fn from(value: Material) -> Self {
            ShaderMtl {
                base_diffuse: value.diffuse().into(),
                base_ambient: value.ambient().into(),
                base_specular: value.specular().into(),
                base_specular_factor: value.base_specular_factor(),
            }
        }
    }
}

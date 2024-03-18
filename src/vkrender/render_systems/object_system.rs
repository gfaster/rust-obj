//! To render an object, we need to do several things:
//! - create a pipeline that has shaders, subpass, and descriptor sets
//! - create a renderpass that contains the subpass as well as attachments
//! - have subpass inputs and outputs bound
//! - potentially have samplers bound
//! - put it all together in a command buffer
//!
//! I'm making separate structs for this instead of a trait because of how different they are
//!
//! Todo:
//! -[ ] Lazy upload of textures if the set render mode doesn't need them

use std::cell::Cell;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
    CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    SecondaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::{input_assembly::InputAssemblyState, viewport::Viewport};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::pipeline::{Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};

use vulkano::shader::EntryPoint;
use vulkano::DeviceSize;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    pipeline::{graphics::viewport::ViewportState, GraphicsPipeline},
    render_pass::Subpass,
};

use crate::mesh::{MeshData, MeshDataBuffs};
use crate::partition;
use crate::util::CellMap;
use crate::{
    controls::Camera,
    mesh::{mtl::Material, MeshMeta},
};

use super::super::VkVertex;

#[derive(Default)]
pub enum ObjectSystemRenderMode {
    #[default]
    Color,

    Depth,
}

#[derive(Default)]
pub struct ObjectSystemConfig {
    pub blend_mode: ColorBlendState,
    pub vertex_arena_size: vulkano::DeviceSize,
    pub index_arena_size: vulkano::DeviceSize,
    pub render_mode: ObjectSystemRenderMode,
}

/// Basic rendering of an object
///
/// ## Usage
/// - create a new `ObjectSystem`
/// - register meshes
/// - draw
///
/// Goal: https://vkguide.dev/docs/gpudriven/gpu_driven_engines/
/// Stretch goal: https://www.nvidia.com/content/GTC-2010/pdfs/2152_GTC2010.pdf
pub struct ObjectSystem {
    objects: Vec<ObjectInstance>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    render_mode: ObjectSystemRenderMode,

    /// command buffer for textures that need to be uploaded to the GPU. uploading happens on draw,
    /// if there are any pending
    upload_cmdbuf: Cell<
        Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, StandardCommandBufferAllocator>>,
    >,
    unloaded_texture_count: Cell<u32>,

    /// various necessary vulkano components
    gfx_queue: Arc<Queue>,
    cmdbuf_allocator: Arc<StandardCommandBufferAllocator>,
    descset_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,

    /// the subbuffer allocator is owned because I think it may be better for reducing
    /// fragmentation
    uniform_buffer: SubbufferAllocator<StandardMemoryAllocator>,
    vertex_buffer: SubbufferAllocator<StandardMemoryAllocator>,
    index_buffer: SubbufferAllocator<StandardMemoryAllocator>,
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

impl ObjectSystem {
    #[inline]
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        dimensions: [f32; 2],
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        Self::new_with_config(
            Default::default(),
            gfx_queue,
            subpass,
            dimensions,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        )
    }

    pub fn new_with_config(
        config: ObjectSystemConfig,
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        extent: [f32; 2],
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        let device = gfx_queue.device().clone();
        let vs = vs::load(device.clone()).unwrap();
        let fs;
        match config.render_mode {
            ObjectSystemRenderMode::Depth => fs = fs::load_depth(device.clone()).unwrap(),
            ObjectSystemRenderMode::Color => fs = fs::load_color(device.clone()).unwrap(),
        }

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let vertex_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                arena_size: config.vertex_arena_size,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let index_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDEX_BUFFER,
                arena_size: config.index_arena_size,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let vs_entry = vs.entry_point("main").unwrap();
        let fs_entry = fs.entry_point("main").unwrap();

        let pipeline = Self::make_pipeline(device, subpass.clone(), vs_entry, fs_entry, extent);

        Self {
            objects: Vec::new(),
            pipeline,
            subpass,
            cmdbuf_allocator: command_buffer_allocator,
            descset_allocator: descriptor_set_allocator,
            memory_allocator,
            gfx_queue,
            uniform_buffer,
            index_buffer,
            vertex_buffer,
            render_mode: config.render_mode,
            upload_cmdbuf: None.into(),
            unloaded_texture_count: 0.into(),
        }
    }

    fn make_pipeline(
        device: Arc<Device>,
        subpass: Subpass,
        vs_entry: EntryPoint,
        fs_entry: EntryPoint,
        extent: [f32; 2],
    ) -> Arc<GraphicsPipeline> {
        debug_assert!(extent[0] > 0.0);
        debug_assert!(extent[1] > 0.0);
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
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
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

    pub fn draw(
        &self,
        cam: &Camera,
    ) -> Arc<SecondaryAutoCommandBuffer<StandardCommandBufferAllocator>> {
        let layout_0 = self.pipeline.layout().set_layouts().get(0).unwrap().clone();
        let layout_1 = self.pipeline.layout().set_layouts().get(1).unwrap().clone();
        let (s_light, s_cam) = generate_uniforms_0(cam);
        let set_0 = PersistentDescriptorSet::new(
            &self.descset_allocator,
            layout_0,
            subbuffer_write_descriptors! {
                self.uniform_buffer,
                (s_cam, vs::CAM_BINDING),
                (s_light, fs::LIGHT_BINDING)
            },
            [],
        )
        .unwrap();
        self.upload_cmdbuf.take().map(|c| {
            log!("uploading {} texture(s)", self.unloaded_texture_count.get());
            c.build().unwrap().execute(self.gfx_queue.clone())
        });
        self.unloaded_texture_count.set(0);

        let sampler = Sampler::new(
            self.gfx_queue.device().clone(),
            SamplerCreateInfo::simple_repeat_linear(),
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
                set_0,
            )
            .unwrap();

        for object in &self.objects {
            let (s_mat, s_mtl) = generate_uniforms_1(&object);
            let s_part = object.partition.clone();
            let set_1 = PersistentDescriptorSet::new(
                &self.descset_allocator,
                layout_1.clone(),
                subbuffer_write_descriptors! {
                    self.uniform_buffer,
                    (s_mat, vs::MAT_BINDING),
                    (s_mtl, fs::MTL_BINDING)
                }
                .into_iter()
                // .chain([WriteDescriptorSet::image_view_sampler(
                //     fs::SAMPLER_BINDING,
                //     object.texture.clone(),
                //     sampler.clone(),
                // ),])
                // .chain([WriteDescriptorSet::buffer(fs::CLUSTER_BINDING, s_part)]),
                ,
                [],
            )
            .unwrap_or_else(|e| panic!("{e:?}"));

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    1,
                    set_1,
                )
                .unwrap()
                .bind_vertex_buffers(0, object.vertices.clone())
                .unwrap()
                .bind_index_buffer(object.indices.clone())
                .unwrap()
                .draw_indexed(object.indices.len() as u32, 1, 0, 0, 0)
                .unwrap();
        }
        builder.build().unwrap()
    }

    pub fn regenerate(&mut self, dimensions: [f32; 2]) {
        let vs = vs::load(self.gfx_queue.device().clone()).unwrap();

        let fs;
        match self.render_mode {
            ObjectSystemRenderMode::Depth => {
                log!("regenerating with depth...");
                fs = fs::load_depth(self.gfx_queue.device().clone()).unwrap()
            }
            ObjectSystemRenderMode::Color => {
                log!("regenerating with color...");
                fs = fs::load_color(self.gfx_queue.device().clone()).unwrap()
            }
        }

        let pipeline = Self::make_pipeline(
            self.gfx_queue.device().clone(),
            self.subpass.clone(),
            vs.entry_point("main").unwrap(),
            fs.entry_point("main").unwrap(),
            dimensions,
        );

        self.pipeline = pipeline;
    }

    pub fn register_object(
        &mut self,
        object: MeshData,
        transform: glm::Mat4,
        do_part: bool,
    ) -> &[ObjectInstance] {
        let part: Option<Arc<[i32]>>;
        if do_part {
            if let Some(succ) = partition::partition_mesh(&object) {
                part = Some(succ.into());
            } else {
                log!("Parition failed");
                part = None
            }
        } else {
            part = None;
        }
        let meta = object.get_meta();
        let mesh_buffs: MeshDataBuffs<VkVertex> = object.into();

        let vertices = self
            .vertex_buffer
            .allocate_slice(mesh_buffs.verts.len() as vulkano::DeviceSize)
            .unwrap();
        let mut indices = self
            .index_buffer
            .allocate_slice(mesh_buffs.indices.len() as vulkano::DeviceSize)
            .unwrap();
        vertices.write().unwrap().copy_from_slice(&mesh_buffs.verts);
        indices
            .write()
            .unwrap()
            .copy_from_slice(&mesh_buffs.indices);

        let part = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            part.unwrap().into_iter().copied(),
        )
        .unwrap();

        let mut upload_cmdbuf = self.upload_cmdbuf.take().unwrap_or_else(|| {
            AutoCommandBufferBuilder::primary(
                &*self.cmdbuf_allocator,
                self.gfx_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap()
        });

        let first_idx = self.objects.len();

        let mut prev = 0;
        for (mtl_idx, material) in meta.materials.iter().enumerate() {
            let img = material
                .1
                .diffuse_map()
                .filter(|i| i.width() > 0 && i.height() > 0)
                .unwrap_or_else(|| Material::dev_texture().into());

            let upload_buf: Subbuffer<[f32]> = Buffer::new_slice(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (img.width() * img.height() * 4) as DeviceSize,
            )
            .unwrap();

            let image = Image::new(
                self.memory_allocator.clone(),
                ImageCreateInfo {
                    format: Format::R8G8B8A8_SRGB,
                    usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                    image_type: ImageType::Dim2d,
                    extent: [img.width(), img.height(), 1],
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            upload_cmdbuf
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buf,
                    image.clone(),
                ))
                .unwrap();

            self.unloaded_texture_count.map_cell(|x| x + 1);

            let mid = material.0 as vulkano::DeviceSize * 3;
            // dbg!(mid - prev);
            let split_indices;
            if (mid - prev) < indices.len() {
                (split_indices, indices) = indices.split_at(mid - prev);
                // (split_indices, indices) = indices.split_at(mid);
            } else {
                split_indices = indices.clone();
            }
            // dbg!(&split_indices.offset());
            // dbg!(&split_indices.len());
            prev = mid;

            self.objects.push(ObjectInstance {
                vertices: vertices.clone(),
                indices: split_indices,
                partition: part.clone(),
                meta: meta.clone(),
                mtl_idx,
                transform,
                texture: ImageView::new_default(image).unwrap(),
            });
        }

        self.upload_cmdbuf = Some(upload_cmdbuf).into();

        &self.objects[first_idx..]
    }

    pub fn register_object_instances(&mut self, instances: &[ObjectInstance]) {
        self.objects.extend_from_slice(instances)
    }
}

#[derive(Clone)]
pub struct ObjectInstance {
    vertices: Subbuffer<[VkVertex]>,
    indices: Subbuffer<[u32]>,
    partition: Subbuffer<[i32]>,
    meta: MeshMeta,
    mtl_idx: usize,
    texture: Arc<ImageView>,
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
        pos: cam.pos,
        transform: cam.get_transform(),
    };

    let light = fs::ShaderLight {
        light_pos: glm::vec3(10.0, 10.0, 0.0),
        ambient_strength: 0.2,
        light_strength: 10.0,
    };

    (light, cam_attr)
}

pub fn generate_uniforms_1(instance: &ObjectInstance) -> (vs::ShaderMatBuffer, fs::ShaderMtl) {
    let meta = &instance.meta;
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

    let mtl = meta.materials[instance.mtl_idx].1.clone().into();

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
        shaders: {
            color: {
                ty: "fragment",
                path: "shaders/frag_color_vk.glsl",
            },
            depth: {
                ty: "fragment",
                path: "shaders/frag_vk.glsl",
            },
        },
        linalg_type: "nalgebra",
        custom_derives: [Debug]
    }

    // TODO: this might be revealed after entry point is added
    pub const MTL_BINDING: u32 = 2;
    pub const LIGHT_BINDING: u32 = 3;
    pub const SAMPLER_BINDING: u32 = 4;
    pub const CLUSTER_BINDING: u32 = 5;

    // declaration of ShaderMtl from macro parsing of frag_vk.glsl
    impl From<Material> for ShaderMtl {
        fn from(value: Material) -> Self {
            ShaderMtl {
                base_diffuse: value.diffuse().into(),
                base_ambient: value.ambient().into(),
                base_specular: value.specular().into(),
                base_specular_factor: value.base_specular_factor(),
                use_sampler: value.diffuse_map().is_some() as u32,
                tri_start: 0,
                use_clusters: true as u32,
            }
        }
    }
}

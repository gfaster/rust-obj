
use vulkano::buffer::{BufferContents, Buffer, BufferCreateInfo, BufferUsage};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{DeviceExtensions, QueueFlags, Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::image::ImageUsage;
use vulkano::instance::Instance;
use vulkano::{VulkanLibrary, format};
use vulkano::memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryUsage};
use vulkano::pipeline::graphics::vertex_input;
use vulkano::swapchain::Swapchain;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{WindowBuilder, self, Window};

use crate::glm::{self, Vec3};
use crate::mesh::{self, MeshData, Vertex, MeshDataBuffs};



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

impl From<Vertex> for VkVertex {
    fn from(value: Vertex) -> Self {
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
            path: "shaders/vert_vk.glsl"
        }
    }
    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/frag_vk.glsl"
        }
    }
}

pub fn depth_screenshots(_m: MeshData, _dim: (u32, u32), _pos: &[Vec3]) -> Vec<String> {
    todo!()
}

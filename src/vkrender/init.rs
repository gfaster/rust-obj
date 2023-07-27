use std::sync::Arc;

use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};

use vulkano::instance::Instance;

use vulkano::swapchain::Surface;

use vulkano::VulkanLibrary;
use vulkano_win::VkSurfaceBuild;

use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

/// initialize the device to use for rendering that supports necessary extensions with a surface.
pub fn initialize_device_window(
    device_extensions: DeviceExtensions,
) -> (Arc<Device>, Arc<Queue>, Arc<Surface>, EventLoop<()>) {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);

    let event_loop = EventLoop::new();

    let instance = Instance::new(
        library,
        vulkano::instance::InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

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
        },
    )
    .unwrap();

    // only have one queue, so we just use that
    let queue = queues.next().unwrap();

    (device, queue, surface, event_loop)
}

/// initialize the device to use for rendering that supports necessary extensions.
pub fn initialize_device(device_extensions: DeviceExtensions) -> (Arc<Device>, Arc<Queue>) {
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

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        // .inspect(|p| eprintln!("{p:#?}"))
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter(|p| p.api_version() >= vulkano::Version::V1_1)
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

    // only have one queue, so we just use that
    let queue = queues.next().unwrap();

    (device, queue)
}

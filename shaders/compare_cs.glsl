#version 450

// #extension GL_KHR_shader_subgroup_basic : enable
// #extension GL_KHR_shader_subgroup_arithmetic : enable
// #extension GL_EXT_shader_atomic_float : enable



// https://stackoverflow.com/a/68592086/7487237
// https://www.khronos.org/blog/vulkan-subgroup-tutorial
// https://dournac.org/info/gpu_sum_reduction

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

// layout(set = 0, binding = 0) readonly buffer ShaderComputeData {
//     float data[];
// } data;

layout(set = 0, binding = 0) uniform sampler2D s_tex;

layout(set = 0, binding = 1) buffer ShaderComputeResult {
    float results[];
} results;


void main() {

    uint start = PIXEL_NUM * gl_GlobalInvocationID.x;
    float sum = 0.0f;
    for (uint i = 0; i < PIXEL_NUM; i++) {
        uint idx = i + start;
        sum += texelFetch(s_tex, ivec2(idx % 1024, idx / 1024), 0).r;
    }
    results.results[gl_GlobalInvocationID.x] = sum;
    // results.results[gl_GlobalInvocationID.x] = 1.0;
    // results.results[gl_GlobalInvocationID.x] = data.data[1];
}

#version 450

#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_atomic_float : enable



// https://stackoverflow.com/a/68592086/7487237
// https://www.khronos.org/blog/vulkan-subgroup-tutorial
// https://dournac.org/info/gpu_sum_reduction

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer ShaderComputeData {
    float data[];
} data;

layout(set = 0, binding = 0) buffer ShaderComputeResult {
    float result;
} result;


void main(){
    float sum = subgroupAdd(data.data[gl_GlobalInvocationID.x * 4]);

    if (subgroupElect()) {
        // oh no. no atomic add for floats - only ints
        atomicAdd(result.result, sum);
    }
}

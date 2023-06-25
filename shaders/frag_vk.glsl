#version 460

layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec3 v_fragPos;
layout(location = 1) in vec3 v_fragNorm;
layout(location = 2) in vec2 v_texCoord;
layout(location = 3) in float v_depth;

// rust structs are generated from this strut name
layout(set = 1, binding = 0) uniform ShaderMtl {
    vec4 base_diffuse;
    vec4 base_ambient;
    vec4 base_specular;
    float base_specular_factor;
} Mtl;

// rust structs are generated from this struct name
layout(set = 2, binding = 0) uniform ShaderLight {
    vec3 light_pos;
} Light;

// subroutine(shading_routine_t) vec4 depth_buffer() {
//     float depth = v_depth * gl_FragCoord.w;
//     return vec4(vec3(depth), 1.0f);
// }

void main()
{
    vec3 base_color = Mtl.base_diffuse.xyz;

    vec3 norm = normalize(v_fragNorm);
    vec3 light_dir = normalize(Light.light_pos - v_fragPos);
    vec3 reflect_dir = reflect(-light_dir, norm);

    float spec = max(dot(norm, reflect_dir), 0.0f);
    float diff = max(dot(norm, light_dir), 0.0f);

    vec3 diff_color = diff * base_color;
    vec3 spec_color = pow(spec, Mtl.base_specular_factor) * Mtl.base_specular.rgb;

    FragColor = vec4(diff_color + spec_color + Mtl.base_ambient.rgb, 1.0f); 
}

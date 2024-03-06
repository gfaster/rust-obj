#version 460

layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec3 v_fragPos;
layout(location = 1) in vec3 v_fragNorm;
layout(location = 2) in vec2 v_texCoord;
layout(location = 3) in float v_depth;
layout(location = 4) in vec3 v_camPos;

// rust structs are generated from this strut name
layout(set = 1, binding = 2) uniform ShaderMtl {
    vec4 base_diffuse;
    vec4 base_ambient;
    vec4 base_specular;
    float base_specular_factor;
    bool use_sampler;
    uint tri_start;
    bool use_clusters;
} Mtl;

// rust structs are generated from this struct name
layout(set = 0, binding = 3) uniform ShaderLight {
    vec3 light_pos;
    float ambient_strength;
    float light_strength;
} Light;

layout(set = 1, binding = 4) uniform sampler2D s_tex;

layout(set = 1, binding = 5) readonly buffer ClusterData {
    uint data[];
} clusterData;


vec4 diffuse(vec3 base_color) {
    vec3 norm = normalize(v_fragNorm);
    vec3 light_dir = normalize(Light.light_pos - v_fragPos);
    vec3 reflect_dir = reflect(-light_dir, norm);

    float spec = max(dot(norm, reflect_dir), 0.0f);
    float diff = max(dot(norm, light_dir), 0.0f);

    vec3 diff_color = diff * base_color;
    vec3 spec_color = pow(spec, Mtl.base_specular_factor) * Mtl.base_specular.rgb;

    return vec4(diff_color + spec_color + Mtl.base_ambient.rgb, 1.0f); 
}

vec4 depth_buffer() {
    float depth = v_depth * gl_FragCoord.w;
    // float depth = 0.5;
    return vec4(vec3(depth), 1.0f);
    // return vec4(vec3(1.0), 1.0);
}

vec4 color_correct(vec4 color) {
    return color;
}

vec3 cluster_color(uint id) {
    /* adopted from this https://stackoverflow.com/a/12996028/7487237 */
    uint x = id;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = (x >> 16) ^ x;

    vec3 ret;
    ret.r = float((x >> 0) % (1u << 8)) / float(1u<<8);
    ret.g = float((x >> 8) % (1u << 8)) / float(1u<<8);
    ret.b = float((x >> 16) % (1u << 8)) / float(1u<<8);

    return ret;
}

void main()
{
    vec4 base_color;
    if (Mtl.use_sampler) {
        base_color = texture(s_tex, v_texCoord);
        if (base_color.w < 0.1) {
            discard;
        }
    } else if (Mtl.use_clusters) {
        base_color = vec4(cluster_color(Mtl.tri_start + gl_PrimitiveID), 1.0f);
    } else {
        base_color = Mtl.base_diffuse;
    }

    FragColor = base_color;
}

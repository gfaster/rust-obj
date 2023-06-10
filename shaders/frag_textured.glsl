#version 330 core
#extension GL_ARB_shader_subroutine : require

out vec4 FragColor;

in vec3 v_fragPos;
in vec3 v_fragNorm;
in vec2 v_texCoord;
in float v_depth;


uniform sampler2D diffuse_map;
//uniform sampler2D u_NormalMap;

uniform vec3 light_pos;
uniform vec4 base_diffuse;
uniform vec4 base_ambient;
uniform vec4 base_specular;
uniform float base_specular_factor;

subroutine vec4 shading_routine_t();
subroutine uniform shading_routine_t shading_routine;

subroutine(shading_routine_t) vec4 depth_buffer() {
    float depth = v_depth * gl_FragCoord.w;
    return vec4(vec3(depth), 1.0f);
}

subroutine(shading_routine_t) vec4 shaded() {
    vec3 base_color = texture(diffuse_map, v_texCoord).rgb;

    vec3 norm = normalize(v_fragNorm);
    vec3 light_dir = normalize(light_pos - v_fragPos);
    vec3 reflect_dir = reflect(-light_dir, norm);

    float spec = max(dot(norm, reflect_dir), 0.0f);
    float diff = max(dot(norm, light_dir), 0.0f);

    vec3 diff_color = diff * base_color * 0.5f;
    vec3 spec_color = pow(spec, base_specular_factor) * base_specular.rgb * 1.0f;
    vec3 ambient_color = base_ambient.rgb * 0.1f;

    return vec4(diff_color + spec_color + ambient_color, 1.0f); 
}

void main()
{
    FragColor = shading_routine();
}

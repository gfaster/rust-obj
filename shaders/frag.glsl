#version 330 core
#extension GL_ARB_shader_subroutine : require

out vec4 FragColor;

in vec3 v_fragPos;
in vec3 v_fragNorm;
in vec2 v_texCoord;
in float v_depth;
in vec4 gl_FragCoord; 


//uniform sampler2D u_DiffuseMap; 
//uniform sampler2D u_NormalMap; 

uniform vec3 light_pos;
uniform gl_DepthRangeParameters gl_DepthRange;

subroutine vec4 shading_routine_t();
subroutine uniform shading_routine_t shading_routine;

subroutine(shading_routine_t) vec4 depth_buffer() {
    float depth = v_depth * gl_FragCoord.w;
    return vec4(vec3(depth), 1.0f);
}

subroutine(shading_routine_t) vec4 shaded() {
    vec3 base_color = vec3(1.0f, 0.0f, 0.0f);
    float spec_strength = 0.1f;
    float ambient_strength = 0.05f;

    vec3 norm = normalize(v_fragNorm);
    vec3 light_dir = normalize(light_pos - v_fragPos);
    vec3 reflect_dir = reflect(-light_dir, norm);

    float spec = max(dot(norm, reflect_dir), 0.0f);
    float diff = max(dot(norm, light_dir), 0.0f);

    vec3 diff_color = diff * base_color;
    vec3 spec_color = pow(spec, 32) * vec3(1.0f) * spec_strength;
    vec3 ambient_color = base_color * ambient_strength;

    return vec4(diff_color + spec_color + ambient_color, 1.0f); 
}

void main()
{
    FragColor = shading_routine();
}

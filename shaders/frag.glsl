#version 330 core

out vec4 FragColor;

in vec3 v_fragPos;
in vec3 v_fragNorm;
in vec2 v_texCoord;


//uniform sampler2D u_DiffuseMap; 
//uniform sampler2D u_NormalMap; 

uniform vec3 light_pos;


void main()
{
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

    FragColor = vec4(diff_color + spec_color + ambient_color, 1.0f); 
}

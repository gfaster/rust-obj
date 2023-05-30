#version 330 core

out vec4 FragColor;

in vec4 v_eyePos;
in vec4 v_eyeNorm;
in vec2 v_texCoord;


//uniform sampler2D u_DiffuseMap; 
//uniform sampler2D u_NormalMap; 

uniform vec4 light_pos;


void main()
{
    vec4 dir = normalize(light_pos);
    vec4 reflect_dir = reflect(-dir, v_eyeNorm);
    vec4 v = normalize(-v_eyePos);
    float spec = max(dot(v, reflect_dir), 0.0f);
    float diff = max(dot(v_eyeNorm, dir), 0.0f);

    vec3 base_color = vec3(1.0f, 0.0f, 0.0f);

    float spec_strength = 1.5f;
    float ambient_strength = 0.025f;

    vec3 diff_color = diff * base_color;
    vec3 spec_color = pow(spec, 25) * vec3(1.0f) * spec_strength;
    vec3 ambient_color = base_color * ambient_strength;

    FragColor = vec4(diff_color + spec_color + ambient_color, 1.0f); 
    // FragColor = vec4(spec_color, 1.0f); 
}

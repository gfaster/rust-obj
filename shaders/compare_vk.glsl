#version 460

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_left;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_right;

layout(location = 0) out vec4 FragColor;

// layout(location = 0) in vec3 v_fragPos;
// layout(location = 1) in vec3 v_fragNorm;
// layout(location = 2) in vec2 v_texCoord;
// layout(location = 3) in float v_depth;


void main()
{

    // FragColor = vec4(v_fragPos, 0.0);
    // FragColor = vec4(v_fragNorm, 0.0);
    // FragColor = vec4(v_texCoord, 0.0, 0.0);
    // FragColor = vec4(v_depth);

    FragColor = abs(subpassLoad(u_left) - subpassLoad(u_right));
}

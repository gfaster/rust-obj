#version 460

// https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_vulkan_glsl.txt

layout(location = 0) in vec3 position; 
// layout(location = 1) in vec3 normal; 
// layout(location = 2) in vec2 tex;


void main()
{
  vec2 trans = (position.xy - 0.5f) * 2.0f;
  vec4 pos = vec4(trans, 0.0f, 1.0f);
  gl_Position = pos;
}

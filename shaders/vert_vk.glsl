#version 460

// https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_vulkan_glsl.txt

layout(location = 0) in vec3 position; 
layout(location = 1) in vec3 normal; 
layout(location = 2) in vec2 tex;

layout(location = 0) out vec3 v_fragPos;
layout(location = 1) out vec3 v_fragNorm;
layout(location = 2) out vec2 v_texCoord;
layout(location = 3) out float v_depth;


layout(set = 0, binding = 0) uniform matBuffer {
  mat4 transform;
  mat4 modelview;
  mat4 projection_matrix;
  mat3 normal_matrix;
} Matrices;

layout(set = 0, binding = 3) uniform camAttrBuffer {
  float near;
  float far;
} CamAttr;

void main()
{
  vec4 pos = vec4(position, 1.0f);

  v_fragNorm = normalize(Matrices.normal_matrix * normal);
  v_fragPos = vec3(Matrices.transform * pos);
  v_texCoord = tex;

  gl_Position = Matrices.projection_matrix * Matrices.modelview * pos;
  v_depth = gl_Position.z / CamAttr.far;
}

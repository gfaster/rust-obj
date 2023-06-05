#version 330 core

in vec3 position; 
in vec3 normal; 
in vec2 tex;

out vec3 v_fragPos;
out vec3 v_fragNorm;
out vec2 v_texCoord;
out float v_depth;

uniform mat4 transform;
uniform mat4 modelview;
uniform mat4 projection_matrix;
uniform mat3 normal_matrix;
uniform gl_DepthRangeParameters gl_DepthRange;

void main()
{
  vec4 pos = vec4(position, 1.0f);

  v_fragNorm = normalize(normal_matrix * normal);
  v_fragPos = vec3(transform * pos);
  v_texCoord = tex;

  gl_Position = projection_matrix * modelview * pos;
  v_depth = gl_Position.z / gl_DepthRange.far;
}

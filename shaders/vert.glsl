#version 330 core

in vec3 position; 
in vec3 normal; 
in vec2 tex;

out vec4 v_eyePos;
out vec4 v_eyeNorm;
out vec2 v_texCoord;

uniform mat4 transform;
uniform mat4 projection_matrix;
uniform mat4 normal_matrix;

void main()
{
  vec4 pos = vec4(position, 1.0f);

  v_eyeNorm = normalize(normal_matrix * vec4(normal, 0.0));
  v_eyePos = transform * pos;
  v_texCoord = tex;


  gl_Position = projection_matrix * transform * pos;
}

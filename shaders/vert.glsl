#version 330 core

// This is adopted from my cs class's vert (and frag) shaders
// It's a very standard setup

in vec3 position; 
in vec3 normal; 
in vec2 tex;

out vec3 FragPos;
out vec2 v_texCoord;

uniform mat4 transform;
uniform mat4 projection_matrix;
uniform vec3 view_pos;

void main()
{
  gl_Position = projection_matrix * transform * vec4(position, 1.0f);

  v_texCoord = tex;
}

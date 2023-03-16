#version 330 core

// This is adopted from my cs class's vert (and frag) shaders
// It's a very standard setup

in vec3 position; 
in vec3 normals; 
in vec2 texCoord;
in vec3 tangents;
in vec3 bitangents;

out vec3 FragPos;
out vec2 v_texCoord;
out vec3 TangentLightPos;
out vec3 TangentViewPos;
out vec3 TangentFragPos;

uniform mat4 modelTransformMatrix;
uniform mat4 projectionMatrix;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
  gl_Position = projectionMatrix * modelTransformMatrix * vec4(position, 1.0f);;

  vec3 T = vec3(vec4(tangents, 1.0f));
  vec3 B = vec3(vec4(bitangents, 1.0f));
  vec3 N = vec3(vec4(normals, 1.0f));
  mat3 TBN = mat3(modelTransformMatrix) * mat3(T, B, N);
  TangentLightPos = TBN * lightPos;
  TangentViewPos = TBN * viewPos;
  TangentFragPos = TBN * vec3(modelTransformMatrix * vec4(position, 1.0f));

  v_texCoord = texCoord;
}

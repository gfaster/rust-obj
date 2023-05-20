#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 v_texCoord;


//uniform sampler2D u_DiffuseMap; 
//uniform sampler2D u_NormalMap; 

void main()
{
	// vec3 normal = texture(u_NormalMap, v_texCoord).rgb;
	// vec3 color =  texture(u_DiffuseMap, v_texCoord).rgb;
	vec3 color = vec3(1.0f, 0.0f, 0.0f);

	float ambient = 0.3f;

	FragColor = vec4(color * ambient, 1.0f);
}

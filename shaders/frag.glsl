#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 v_texCoord;
in vec3 TangentLightPos;
in vec3 TangentViewPos;
in vec3 TangentFragPos;


uniform sampler2D u_DiffuseMap; 
uniform sampler2D u_NormalMap; 

void main()
{
	vec3 normal = texture(u_NormalMap, v_texCoord).rgb;
	vec3 color =  texture(u_DiffuseMap, v_texCoord).rgb;

	vec3 c_normal = normalize(normal * 2.0f - 1.0f);
	vec3 lightDir = normalize(TangentLightPos - TangentFragPos);

	float diffuse = max(dot(lightDir, c_normal), 0.0f);
	float ambient = 0.1f;

	FragColor = vec4(color * (ambient + diffuse), 1.0f);
}

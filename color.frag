#version 330 core
struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

Light light;
Material material;
in vec3 Normal;
in vec3 FragPos;
in vec3 viewPos;
in vec3 ambient;
in vec3 diffuse;
in vec3 specular;
in float shininess;
out vec4 FragColor;

uniform vec3 objectColor = vec3(1.0, 0.5, 0.31);
uniform vec3 lightColor = vec3(0.5, 0.5, 0.5);
uniform vec3 lightPos = vec3(1.2, 1.0, 2.0);



void main()
{
    light.ambient = vec3(0.2, 0.2, 0.2);
light.diffuse = vec3(0.5, 0.5, 0.5);
light.specular = vec3(1.0, 1.0, 1.0);
light.position = vec3(1.2, 1.0, 2.0);
    material.ambient = ambient;
    material.diffuse = diffuse;
    material.specular = specular;
    material.shininess = shininess;

    // ambient
    vec3 ambient = light.ambient * material.ambient;

    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * material.diffuse);

    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}
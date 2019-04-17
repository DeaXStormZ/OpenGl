#version 330 core
layout(location = 0) in vec3 position;
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
void main() {
//    gl_Position = vec4(position, 1);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    Normal = aNormal;
}
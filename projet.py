#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os  # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL  # standard Python OpenGL wrapper
import glfw  # lean window system wrapper for OpenGL
import numpy as np  # all matrix manipulations & OpenGL args
import pyassimp  # 3D resource loader
import pyassimp.errors  # Assimp error management + exceptions

from transform import *
from PIL import Image  # load images for textures
from itertools import cycle
from transform import lerp, vec
from bisect import bisect_left  # search sorted keyframe lists


# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """

    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i + 1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:  # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""

    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers += [GL.glGenBuffers(1)]
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


# ------------  simple color fragment shader demonstrated in Practical 1 ------
COLOR_VERT = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = aNormal;
} """

COLOR_FRAG = """#version 330 core
in vec3 Normal;
in vec3 FragPos;
out vec4 FragColor;

uniform vec3 objectColor = vec3(1.0, 0.5, 0.31);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 lightPos = vec3(1.2, 1.0, 2.0);

void main()
{
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}"""

PHONG_VERT = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 Normal;
out vec3 FragPos;
out vec3 viewPos;
out vec3 ambient;
out vec3 diffuse;
out vec3 specular;
out float shininess;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    viewPos = aPos;
    gl_Position = projection * view * vec4(FragPos, 1.0);

} """
# TODO check the def of viewpos cf https://learnopengl.com/Lighting/Basic-Lighting, and change the decl of ambient, diffuse and specular like
# uniform Material material and pass in python the vec3
PHONG_FRAG = """#version 330 core
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
uniform vec3 ambient;
uniform vec3 diffuse;
uniform vec3 specular;
uniform float shininess;
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
    
    vec3 result = ambient + diffuse;
    FragColor = vec4(result, 1.0);
}"""

# -------------- Example texture plane class ----------------------------------
TEXTURE_VERT = """#version 330 core
uniform mat4 modelviewprojection;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 tex_position;
out vec2 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    fragTexCoord = tex_position.xy;
}"""

TEXTURE_FRAG = """#version 330 core
uniform sampler2D diffuseMap;
in vec2 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
}"""

# ------------  simple color fragment shader demonstrated in Practical 1 ------
SHINING_VERT = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = aNormal;
} """

SHINING_FRAG = """#version 330 core
in vec3 Normal;
in vec3 FragPos;
out vec4 FragColor;

uniform vec3 objectColor = vec3(1.0, 0.5, 0.31);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 lightPos = vec3(1.2, 1.0, 2.0);

void main()
{
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}"""


# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """

    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            tex = np.asarray(Image.open(file).convert('RGBA'))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


# ------------  Scene object classes ------------------------------------------
class Node:
    """ Scene graph transform and parameter broadcast node """

    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        model = model @ self.transform  # TODO not sure what to insert here for hierarchical update?
        for child in self.children:
            child.draw(projection, view, model, **param)


class TexturedPlane:
    """ Simple first textured object """

    def __init__(self, file):
        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # triangle and face buffers
        vertices = 100 * np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
        faces = np.array(((0, 1, 2), (0, 2, 3)), np.uint32)
        self.vertex_array = VertexArray([vertices], faces)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = file

        # setup texture and upload it to GPU
        self.texture = Texture(file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.execute(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


# mesh to refactor all previous classes
class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader, **param):
        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(GL.GL_TRIANGLES)


class PhongMesh:
    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, phong_shader=None, Kd=(1, 1, 1),
             Ka=(0, 0, 0), Ks=(1, 1, 1), Ns=16.0, **params):
        names = ['view', 'projection', 'model', 'diffuse', 'ambient', 'specular',
                 'shininess']  # TODO prend le Ns par default alors qu il y en a un dans le mtl
        loc = {n: GL.glGetUniformLocation(phong_shader.glid, n) for n in names}
        GL.glUseProgram(phong_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)
        GL.glUniform3fv(loc['diffuse'], 1, Kd)
        GL.glUniform3fv(loc['ambient'], 1, Ka)
        GL.glUniform3fv(loc['specular'], 1, Ks)
        GL.glUniform1f(loc['shininess'], Ns)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(GL.GL_TRIANGLES)


class ShiningMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, shining_shader=None, **param):
        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(shining_shader.glid, n) for n in names}
        GL.glUseProgram(shining_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(GL.GL_TRIANGLES)


class TexturedMesh:
    def __init__(self, file, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)
        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)

        # setup texture and upload it to GPU
        self.file = file

        # setup texture and upload it to GPU
        self.texture = Texture(file, self.wrap_mode, *self.filter_mode)
        self.shader = None

    def draw(self, projection, view, model, texture_shader=None, win=None, **_kwargs):
        self.shader = texture_shader
        # some interactive elements
        if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.execute(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


class SimpleTriangle(ColorMesh):
    """Hello triangle object"""

    def __init__(self):
        # triangle position buffer
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        super().__init__([position, color])


class Saturn(Node):
    def __init__(self):
        super().__init__()
        self.add(*load_textured('Saturn/13906_Saturn_v1_l3.obj'))


class Spaceship(Node):
    def __init__(self):
        super().__init__()
        self.add(*load('spaceship/ship.obj'))  # just load the obj from file


class WormHole(Node):
    def __init__(self):
        super().__init__()
        self.add(*load_textured('Jupiter/jupiter.obj'))


class Bullet(Node):
    def __init__(self):
        super().__init__()
        self.add(*load_shining('bullet/bullet.obj'))  # just load the obj from file


# -------------- 3D ressource loader -----------------------------------------

def load_shining(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # reorganize and keep only first material properties of each
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))

    # prepare mesh nodes
    meshes = []
    for mesh in scene.meshes:
        mat = scene.materials[mesh.materialindex].tokens
        node = Node(Kd=mat.get('diffuse', (1, 1, 1)),
                    Ka=mat.get('ambient', (0, 0, 0)),
                    Ks=mat.get('specular', (1, 1, 1)),
                    s=mat.get('shininess', 16.))
        node.add(ShiningMesh([mesh.vertices, mesh.normals], mesh.faces))
        meshes.append(node)

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # reorganize and keep only first material properties of each
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))

    # prepare mesh nodes
    meshes = []
    for mesh in scene.meshes:
        mat = scene.materials[mesh.materialindex].tokens
        node = Node(Kd=mat.get('diffuse', (1, 1, 1)),
                    Ka=mat.get('ambient', (0, 0, 0)),
                    Ks=mat.get('specular', (1, 1, 1)),
                    s=mat.get('shininess', 16.))
        node.add(PhongMesh([mesh.vertices, mesh.normals], mesh.faces))
        meshes.append(node)

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


# -------------- 3D textured mesh loader ---------------------------------------
def load_textured(file):
    """ load resources using pyassimp, return list of TexturedMeshes """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    path = os.path.join('.', '') if path == '' else path
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = tname[0]
            else:
                print('Failed to find texture:', tname)

    # prepare textured mesh
    meshes = []
    for mesh in scene.meshes:
        texture = scene.materials[mesh.materialindex].texture

        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)

        # create the textured mesh object from texture, attributes, and indices
        meshes.append(TexturedMesh(texture, [mesh.vertices, tex_uv], mesh.faces))

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""

    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time >= self.times[-1]:
            return self.values[-1]
        elif time <= self.times[0]:
            return self.values[0]
        # 2. search for closest index entry in self.times, using bisect_left function
        index = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        return self.interpolate(self.values[index - 1], self.values[index],
                                (time - self.times[index - 1]) / (self.times[index] - self.times[index - 1]))
        # TODO fix calculation error


# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__(radians=[1, .5, -.6])
        self.distance = 900
        self.pos2d = (0, 0)
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_window_size_callback(self.win, self.on_size)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)  # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)  # backface culling enabled (TP2)

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
        self.phong_shader = Shader(PHONG_VERT, PHONG_FRAG)
        self.texture_shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)
        self.shining_shader = Shader(SHINING_VERT, SHINING_FRAG)

        # initially empty list of object to draw
        self.drawables = []

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    def on_size(self, win, width, height):
        """ window size update => update viewport to new framebuffer size """
        GL.glViewport(0, 0, *glfw.get_framebuffer_size(win))

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(),
                              color_shader=self.color_shader, phong_shader=self.phong_shader,
                              texture_shader=self.texture_shader, shining_shader=self.shining_shader, win=self.win)
            # TODO use texture_shader
            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_SPACE:
                glfw.set_time(0)


class RotationControlNode(Node):
    def __init__(self, key_up, key_down, axis, angle=0, transform_before=identity(),
                 transform_after=identity(), precision=1.0,
                 **param):  # key_right, key_left, position=0, precision_position=0.2,
        super().__init__(**param)  # forward base constructor named arguments
        self.angle, self.axis = angle, axis
        self.key_up, self.key_down = key_up, key_down
        # self.key_right, self.key_left = key_right, key_left
        self.transform_before, self.transform_after = transform_before, transform_after
        self.precision = precision
        # self.position = position
        # self.precision_position = precision_position

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None
        self.angle += self.precision * int(glfw.get_key(win, self.key_up) == glfw.PRESS)
        self.angle -= self.precision * int(glfw.get_key(win, self.key_down) == glfw.PRESS)
        # print(self.angle)
        # self.position += self.precision_position * int(glfw.get_key(win, self.key_right) == glfw.PRESS)
        # self.position -= self.precision_position * int(glfw.get_key(win, self.key_left) == glfw.PRESS)
        # print(self.position)
        # translation = self.position * self.position_axis
        self.transform = rotate((1, 0, 0), min(-100 + np.sqrt(1000*glfw.get_time()), -1)) @ self.transform_before @ \
                         rotate(self.axis, self.angle) @ self.transform_after  # @ translate(0, self.position,0)

        # call Node's draw method to pursue the hierarchical tree calling
        super().draw(projection, view, model, win=win, **param)


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """

    viewer = Viewer()

    # my_keyframes = KeyFrames({0: 1, 3: 7, 6: 20})
    # my_keyframes.value(glfw.get_time())

    # node = RotationControlNode(glfw.KEY_LEFT, glfw.KEY_RIGHT, vec(0, 1, 0), transform_after=translate(360, 0, 360) @ scale(0.2))

    saturn = Node(transform=rotate((0, 1, 0), -72))
    saturn.add(Saturn())

    wormhole = Node(transform=translate(350, 0, 350))
    wormhole.add(WormHole())

    glfw.set_time(0)

    spaceship = RotationControlNode(glfw.KEY_RIGHT, glfw.KEY_LEFT, vec(1, 0, 0),
                                    precision=2,
                                    transform_before=translate(350, 0, 350) @ scale(0.2) @ rotate((0, 0, 1), 90),
                                    transform_after=translate(0, -9, 0))
    spaceship.add(Spaceship())

    bullet = RotationControlNode(glfw.KEY_RIGHT, glfw.KEY_LEFT, vec(1, 0, 0),
                                 precision=2,
                                 transform_before=translate(350, 0, 350) @ scale(0.2) @ rotate((0, 0, 1), 90),
                                 transform_after=rotate((0, 0, 1), 90) @ translate(8, 5, 0))

    bullet1 = RotationControlNode(glfw.KEY_RIGHT, glfw.KEY_LEFT, vec(1, 0, 0),
                                  precision=2,
                                  transform_before=translate(350, 0, 350) @ scale(0.2) @ rotate((0, 0, 1), 90),
                                  transform_after=rotate((0, 0, 1), 90) @ translate(-8, 5, 0))

    bullet2 = RotationControlNode(glfw.KEY_RIGHT, glfw.KEY_LEFT, vec(1, 0, 0),
                                  precision=2,
                                  transform_before=translate(350, 0, 350) @ scale(0.2) @ rotate((0, 0, 1), 90),
                                  transform_after=rotate((0, 0, 1), 90) @ translate(0, 5, -8))

    bullet3 = RotationControlNode(glfw.KEY_RIGHT, glfw.KEY_LEFT, vec(1, 0, 0),
                                  precision=2,
                                  transform_before=translate(350, 0, 350) @ scale(0.2) @ rotate((0, 0, 1), 90),
                                  transform_after=rotate((0, 0, 1), 90) @ translate(0, 5, 8))

    bullet.add(Bullet())
    bullet1.add(Bullet())
    bullet2.add(Bullet())
    bullet3.add(Bullet())

    transform_bullet = Node()
    transform_bullet.add(bullet, bullet1, bullet2, bullet3)

    transform_spaceship = Node()
    transform_spaceship.add(spaceship, transform_bullet)

    transform_wormhole = Node()
    transform_wormhole.add(wormhole, transform_spaceship)

    transform_saturn = Node()
    transform_saturn.add(saturn, transform_wormhole)

    viewer.add(transform_saturn)
    print('the window is resizable')
    print('time can be set to 0 with key_space')
    print('spaceship can rotate on itself with key_right and key_left')
    viewer.run()


if __name__ == '__main__':
    glfw.init()  # initialize window system glfw
    main()  # main function keeps variables locally scoped
    glfw.terminate()  # destroy all glfw windows and GL contexts

#version 330

// World viewer water: texcoord x carries the water depth at the vertex. The surface is lifted a
// little more the farther it is, so shallow water does not z-fight the bed just below it.

in vec3 vertexPosition;
in vec2 vertexTexCoord;

uniform mat4 matModel;
uniform mat4 matView;
uniform mat4 matProjection;
uniform vec3 viewPos;
uniform float lift;          // 1 for patch water; 0 for the open sea, which has no bed to fight

out vec3 fragPosition;
out float fragDepth;

void main()
{
    vec4 world = matModel*vec4(vertexPosition, 1.0);
    world.y += lift*(0.03 + 0.0008*distance(viewPos, world.xyz));
    fragPosition = world.xyz;
    fragDepth = vertexTexCoord.x;
    gl_Position = matProjection*matView*world;
}

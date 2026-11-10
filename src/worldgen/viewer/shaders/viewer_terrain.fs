#version 330

// World viewer terrain: the patch colour texture lit by one sun and ambient light, fading into fog.

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;     // white unless the mesh has colours (towns)

uniform sampler2D texture0;
uniform vec4 colDiffuse;

uniform vec3 sunDir;        // towards the sun
uniform vec3 sunColor;
uniform vec3 ambientColor;
uniform vec3 viewPos;
uniform vec3 fogColor;
uniform vec2 fogRange;      // distance where fog starts, and where it is complete
uniform vec4 tint;          // rgb mixed in by a (debug: colour by level of detail)
uniform float borderWidth;  // in texture coordinates; 0 hides patch borders

out vec4 finalColor;

// Town meshes address a strip of atlasTiles square tiles (r.world.towns.atlas): texcoord x is
// tile*256 + 128 + u and y is v, with u and v counting repeats of the tile. 0 means a plain texture.
uniform float atlasTiles;
const float ATLAS_STRIDE = 256.0;
const float TILE_PIXELS = 32.0;

vec4 albedoAt(vec2 uv)
{
    if (atlasTiles <= 0.0) return texture(texture0, uv);
    float tile = floor(uv.x/ATLAS_STRIDE);
    vec2 local = vec2(uv.x - tile*ATLAS_STRIDE, uv.y);
    vec2 inTile = (0.5 + fract(local)*(TILE_PIXELS - 1.0))/TILE_PIXELS;  // half a texel in, so tiles never bleed
    vec2 scale = vec2(1.0/atlasTiles, 1.0);
    // Gradients of the unwrapped coordinates, so mipmaps do not jump at the seams
    return textureGrad(texture0, (vec2(tile, 0.0) + inTile)*scale, dFdx(local)*scale, dFdy(local)*scale);
}

void main()
{
    vec3 albedo = albedoAt(fragTexCoord).rgb*colDiffuse.rgb*fragColor.rgb;
    float diffuse = max(dot(normalize(fragNormal), normalize(sunDir)), 0.0);
    vec3 color = albedo*(ambientColor + sunColor*diffuse);
    color = mix(color, tint.rgb*(0.4 + 0.6*diffuse), tint.a);

    if (borderWidth > 0.0 && atlasTiles <= 0.0)
    {
        vec2 edge = min(fragTexCoord, 1.0 - fragTexCoord);
        if (min(edge.x, edge.y) < borderWidth) color = mix(color, vec3(1.0, 0.85, 0.2), 0.75);
    }

    float fog = smoothstep(fogRange.x, fogRange.y, distance(viewPos, fragPosition));
    finalColor = vec4(mix(color, fogColor, fog), 1.0);
}

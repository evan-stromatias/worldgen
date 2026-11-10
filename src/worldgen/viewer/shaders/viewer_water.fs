#version 330

// World viewer water: translucent, shallow to deep by depth, with a sun glint and fog.

in vec3 fragPosition;
in float fragDepth;

uniform vec3 shallowColor;
uniform vec3 deepColor;
uniform float deepAt;       // depth where the colour is fully deep
uniform vec3 sunDir;        // towards the sun
uniform vec3 sunColor;
uniform vec3 viewPos;
uniform vec3 fogColor;
uniform vec2 fogRange;

out vec4 finalColor;

void main()
{
    float t = clamp(fragDepth/deepAt, 0.0, 1.0);
    vec3 color = mix(shallowColor, deepColor, t);
    float alpha = mix(0.45, 0.88, t);

    vec3 toEye = normalize(viewPos - fragPosition);
    vec3 halfway = normalize(toEye + normalize(sunDir));
    float glint = pow(max(halfway.y, 0.0), 120.0);
    color += sunColor*glint*0.6;

    float fog = smoothstep(fogRange.x, fogRange.y, distance(viewPos, fragPosition));
    finalColor = vec4(mix(color, fogColor, fog), mix(alpha, 1.0, fog));
}

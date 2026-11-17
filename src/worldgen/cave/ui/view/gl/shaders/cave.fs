#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

uniform vec3 ambientColor;
uniform vec3 lightPos;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform float lightCutoff;
uniform float isFps;

out vec4 finalColor;

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 albedo = fragColor.rgb;

    if (isFps > 0.5) {
        // FPS Flashlight attached to camera
        vec3 toLight = lightPos - fragPosition;
        float dist = length(toLight);
        vec3 L = normalize(toLight);
        float diff = max(dot(N, L), 0.0);
        float atten = 1.0 / (1.0 + 0.03 * dist + 0.001 * dist * dist);
        float cosTheta = dot(-L, normalize(lightDir));
        float spot = clamp((cosTheta - lightCutoff) / (1.0 - lightCutoff), 0.0, 1.0);
        vec3 lit = albedo * (ambientColor + lightColor * diff * atten * spot);
        finalColor = vec4(lit, fragColor.a);
    } else {
        // Orbit directional sun + ambient
        vec3 L = normalize(lightDir);
        float diff = max(dot(N, L), 0.0);
        vec3 lit = albedo * (ambientColor + lightColor * diff);
        finalColor = vec4(lit, fragColor.a);
    }
}

#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

uniform vec3 ambientColor;
uniform vec3 lightDir;
uniform vec3 lightColor;

out vec4 finalColor;

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 albedo = fragColor.rgb;

    // Primary directional light bringing out relief and contours
    vec3 L1 = normalize(lightDir);
    float diff1 = max(dot(N, L1), 0.0);

    // Secondary fill light from opposite angle to illuminate ceilings, undercuts and pillars
    vec3 L2 = normalize(vec3(-lightDir.x, -lightDir.y * 0.5, -lightDir.z));
    float diff2 = max(dot(N, L2), 0.0) * 0.4;

    vec3 lit = albedo * (ambientColor + lightColor * diff1 + vec3(0.5, 0.5, 0.55) * diff2);
    finalColor = vec4(lit, fragColor.a);
}

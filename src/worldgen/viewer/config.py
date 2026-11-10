"""Terrain renderer settings: VIEWER_* environment variables or .env, validated at startup.

Settings describe the rendering only; the terrain comes from the generator.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

type RGB = tuple[float, float, float]


class ViewerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIEWER_", env_file=".env", extra="ignore")

    sky_color: RGB = Field(default=(0.66, 0.78, 0.90), description="Clear colour, also the fog colour (RGB, 0-1).")
    sun_direction: RGB = Field(default=(-0.45, 0.75, -0.5), description="Direction towards the sun (x, y, z).")
    sun_color: RGB = Field(default=(1.0, 0.95, 0.85), description="Direct sunlight (RGB, 0-1).")
    ambient_color: RGB = Field(default=(0.38, 0.42, 0.50), description="Fill light (RGB, 0-1).")

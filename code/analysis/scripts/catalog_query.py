"""
Stage 2: STAC catalog query for Sentinel-2 L2A scenes over Strait of Gibraltar.

Purpose:
    Query the Copernicus Data Space Ecosystem Sentinel Hub Catalog (STAC)
    for available Sentinel-2 Level-2A scenes matching:
      - Geographic area: Strait of Gibraltar
      - Time window: Summer 2025 (June 1 - August 31)
      - Cloud cover: less than 10%

    No data is downloaded; only metadata is retrieved and printed.
    This stage establishes which scenes are available for the pilot.

Usage:
    python catalog_query.py

Author: Rajesh Persad
"""

from pathlib import Path
import sys
from datetime import datetime

from sentinelhub import (
    SHConfig,
    SentinelHubCatalog,
    DataCollection,
    BBox,
    CRS,
)


CREDENTIALS_PATH = Path.home() / ".copernicus_credentials"

# Copernicus Data Space Ecosystem endpoints.
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"

# Strait of Gibraltar bounding box (WGS84 / EPSG:4326).
# Covers the strait itself, port of Algeciras (north shore), Tangier area (south shore),
# and a portion of the western Mediterranean and eastern Atlantic approaches.
# Format: (min_longitude, min_latitude, max_longitude, max_latitude)
GIBRALTAR_BBOX = BBox(bbox=(-5.6, 35.8, -5.0, 36.2), crs=CRS.WGS84)

# Time window for the pilot acquisition.
# Summer 2025 chosen for low cloud probability in the Mediterranean basin.
TIME_INTERVAL = ("2025-06-01", "2025-08-31")

# Cloud cover threshold (percentage of scene covered by clouds).
# 10% chosen as a balance between clean imagery and dataset size.
MAX_CLOUD_COVER = 10.0


def load_credentials(path: Path) -> tuple[str, str]:
    """Read client_id and client_secret from a KEY=VALUE formatted file."""
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found at {path}.")

    creds: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        creds[key.strip()] = value.strip()

    client_id = creds.get("COPERNICUS_CLIENT_ID")
    client_secret = creds.get("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Missing required credentials in file.")
    return client_id, client_secret


def build_config(client_id: str, client_secret: str) -> SHConfig:
    """Build an SHConfig pointed at Copernicus Data Space."""
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = CDSE_TOKEN_URL
    config.sh_base_url = CDSE_BASE_URL
    return config


def query_catalog(config: SHConfig) -> list[dict]:
    """
    Query the STAC catalog for matching Sentinel-2 L2A scenes.

    Returns a list of scene metadata dictionaries, each containing
    id, datetime, cloud cover, and Sentinel-2 tile identifier.
    """
    catalog = SentinelHubCatalog(config=config)

    # Build the STAC filter for cloud cover via CQL2 syntax.
    # The eo:cloud_cover field on each scene is checked against the threshold.
    search_iterator = catalog.search(
        collection=DataCollection.SENTINEL2_L2A,
        bbox=GIBRALTAR_BBOX,
        time=TIME_INTERVAL,
        filter=f"eo:cloud_cover < {MAX_CLOUD_COVER}",
        fields={
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover",
                "properties.grid:code",  # Sentinel-2 tile identifier (e.g., MGRS:30STF)
            ],
            "exclude": [],
        },
    )

    scenes = list(search_iterator)
    return scenes


def summarize_scenes(scenes: list[dict]) -> None:
    """Print a human-readable summary of matching scenes."""
    if not scenes:
        print("No scenes found matching criteria.")
        print("Consider widening the bounding box, time range, or cloud threshold.")
        return

    print(f"\nFound {len(scenes)} matching scenes:\n")
    print(f"{'Date':<20} {'Tile':<15} {'Cloud %':>10} {'Scene ID'}")
    print("-" * 100)

    # Sort by datetime ascending for chronological listing
    scenes_sorted = sorted(scenes, key=lambda s: s["properties"]["datetime"])

    for scene in scenes_sorted:
        scene_id = scene["id"]
        props = scene["properties"]

        # Datetime arrives as ISO format string; trim to date+time for display
        dt_str = props["datetime"][:19].replace("T", " ")
        cloud_pct = props.get("eo:cloud_cover", -1)
        tile = props.get("grid:code", "unknown")

        # Strip the "MGRS-" prefix from tile if present, for compactness
        if tile.startswith("MGRS-"):
            tile = tile[5:]

        print(f"{dt_str:<20} {tile:<15} {cloud_pct:>9.2f}% {scene_id}")


def main() -> int:
    print("Copernicus Data Space STAC catalog query")
    print("=" * 100)
    print(f"  Area: Strait of Gibraltar")
    print(f"  Bounding box: {GIBRALTAR_BBOX}")
    print(f"  Time interval: {TIME_INTERVAL[0]} to {TIME_INTERVAL[1]}")
    print(f"  Max cloud cover: {MAX_CLOUD_COVER}%")
    print(f"  Collection: Sentinel-2 L2A")

    # Load credentials
    try:
        client_id, client_secret = load_credentials(CREDENTIALS_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"FAILED to load credentials: {e}")
        return 1

    # Build config
    config = build_config(client_id, client_secret)

    # Query the catalog
    print("\nQuerying STAC catalog...")
    try:
        scenes = query_catalog(config)
    except Exception as e:
        print(f"FAILED query: {type(e).__name__}: {e}")
        return 1

    # Summarize results
    summarize_scenes(scenes)

    print("\n" + "=" * 100)
    print(f"Query complete: {len(scenes)} scene(s) matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

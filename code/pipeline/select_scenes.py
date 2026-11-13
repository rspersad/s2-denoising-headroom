"""
Stage 3: Scene selection for the pilot acquisition.

Purpose:
    Query the Copernicus Data Space STAC catalog for Sentinel-2 L2A scenes
    over the Strait of Gibraltar in summer 2025, filter to a single tile
    (T30STF) for experimental consistency, and select N scenes distributed
    across time buckets by lowest cloud cover.

    This is the selection logic; downloads happen in Stage 4. The selected
    scenes are written to a manifest CSV that Stage 4 will consume.

Selection methodology:
    1. Query returns all scenes matching the bounding box, time, and cloud
       cover thresholds (same as catalog_query.py).
    2. Filter to scenes from tile T30STF only. This tile is centered on the
       Strait of Gibraltar and contains the densest vessel traffic in the
       region. Limiting to one tile controls for geographic variation,
       allowing the pilot to focus on temporal and noise-injection variation
       as the experimental factors.
    3. Divide the time window into N equal buckets.
    4. From each bucket, select the scene with the lowest cloud cover.
    5. If a bucket contains no scenes, skip it (final selection may have
       fewer than N scenes).

Usage:
    python select_scenes.py

Outputs:
    - Prints selected scenes to stdout.
    - Writes a manifest CSV to ~/praxis/data/manifests/sentinel2_pilot_selection.csv
      containing the metadata needed by Stage 4 for downloading.

Author: Rajesh Persad
"""

from pathlib import Path
import sys
import csv
from datetime import datetime, timedelta

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
GIBRALTAR_BBOX = BBox(bbox=(-5.6, 35.8, -5.0, 36.2), crs=CRS.WGS84)

# Time window and cloud threshold (matches catalog_query.py).
TIME_INTERVAL = ("2025-06-01", "2025-08-31")
MAX_CLOUD_COVER = 10.0

# Tile selection: T30STF is the central Strait of Gibraltar Sentinel-2 tile.
# It contains the densest vessel traffic and provides geographic consistency
# across the pilot dataset.
TARGET_TILE = "T30STF"

# Number of scenes to select for the pilot.
N_SCENES = 10

# Output path for the selection manifest. Lives outside the repo because it
# is a generated artifact, not source. The manifests/ directory under
# /data/praxis/ is the standard location for selection records.
MANIFEST_PATH = Path("/data/praxis/manifests/sentinel2_pilot_selection.csv")


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
    """Query STAC catalog for Sentinel-2 L2A scenes matching pilot criteria."""
    catalog = SentinelHubCatalog(config=config)
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
            ],
            "exclude": [],
        },
    )
    return list(search_iterator)


def extract_tile_from_scene_id(scene_id: str) -> str:
    """
    Extract the Sentinel-2 tile identifier from a scene ID.

    Sentinel-2 scene IDs follow the pattern:
        S2{A|B|C}_MSIL2A_{DATETIME}_N{PROC}_R{ORBIT}_T{TILE}_{PROCDATE}.SAFE

    The tile appears as T{TILE} between the orbit and processing date fields.
    Example: S2C_MSIL2A_20250606T105641_N0511_R094_T30STF_20250606T140657.SAFE
             yields T30STF.
    """
    parts = scene_id.split("_")
    for part in parts:
        if part.startswith("T") and len(part) == 6 and part[1:3].isdigit():
            return part
    return "unknown"


def filter_to_tile(scenes: list[dict], target_tile: str) -> list[dict]:
    """Keep only scenes belonging to the target tile."""
    filtered = []
    for scene in scenes:
        tile = extract_tile_from_scene_id(scene["id"])
        if tile == target_tile:
            scene["_tile"] = tile  # Cache for later use
            filtered.append(scene)
    return filtered


def parse_scene_datetime(scene: dict) -> datetime:
    """Parse the ISO datetime from a scene's properties."""
    dt_str = scene["properties"]["datetime"]
    # Handle the trailing Z if present
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


def select_by_time_buckets(scenes: list[dict], n_buckets: int) -> list[dict]:
    """
    Distribute scenes across N equal time buckets and select the scene
    with lowest cloud cover from each bucket.

    Buckets that contain no scenes are silently skipped, so the returned
    list may have fewer than n_buckets entries.
    """
    if not scenes:
        return []

    # Parse all datetimes
    for scene in scenes:
        scene["_datetime"] = parse_scene_datetime(scene)

    # Get the time window from the actual data (more accurate than constants
    # since query results may not span the full requested range).
    start_time = min(s["_datetime"] for s in scenes)
    end_time = max(s["_datetime"] for s in scenes)
    total_seconds = (end_time - start_time).total_seconds()
    bucket_seconds = total_seconds / n_buckets

    selected = []
    for i in range(n_buckets):
        bucket_start = start_time + timedelta(seconds=i * bucket_seconds)
        bucket_end = start_time + timedelta(seconds=(i + 1) * bucket_seconds)

        # Final bucket includes the end time
        if i == n_buckets - 1:
            in_bucket = [s for s in scenes if bucket_start <= s["_datetime"] <= bucket_end]
        else:
            in_bucket = [s for s in scenes if bucket_start <= s["_datetime"] < bucket_end]

        if not in_bucket:
            continue

        # Choose scene with lowest cloud cover in this bucket
        best = min(in_bucket, key=lambda s: s["properties"]["eo:cloud_cover"])
        selected.append(best)

    return selected


def write_manifest(scenes: list[dict], path: Path) -> None:
    """Write selected scenes to a CSV manifest for Stage 4 to consume."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_id",
            "datetime",
            "tile",
            "cloud_cover_pct",
            "selection_timestamp",
        ])
        selection_ts = datetime.now().isoformat()
        for scene in scenes:
            writer.writerow([
                scene["id"],
                scene["properties"]["datetime"],
                scene.get("_tile", "unknown"),
                scene["properties"]["eo:cloud_cover"],
                selection_ts,
            ])


def print_summary(selected: list[dict], all_tile_scenes: int, all_scenes: int) -> None:
    """Print a human-readable summary of the selection."""
    print(f"\n{'='*100}")
    print(f"Selection summary:")
    print(f"  Total scenes matching query (all tiles):  {all_scenes}")
    print(f"  Filtered to tile {TARGET_TILE}:          {all_tile_scenes}")
    print(f"  Selected for pilot:                       {len(selected)} (target: {N_SCENES})")
    print(f"{'='*100}\n")

    if not selected:
        print("No scenes selected. Check query parameters.")
        return

    print(f"Selected scenes (lowest cloud per time bucket):\n")
    print(f"{'Date':<20} {'Tile':<10} {'Cloud %':>10}  {'Scene ID'}")
    print("-" * 120)

    selected_sorted = sorted(selected, key=lambda s: s["_datetime"])
    for scene in selected_sorted:
        dt_str = scene["properties"]["datetime"][:19].replace("T", " ")
        cloud_pct = scene["properties"]["eo:cloud_cover"]
        tile = scene.get("_tile", "unknown")
        print(f"{dt_str:<20} {tile:<10} {cloud_pct:>9.2f}%  {scene['id']}")


def main() -> int:
    print("Sentinel-2 pilot scene selection")
    print("=" * 100)
    print(f"  Bounding box:    Strait of Gibraltar")
    print(f"  Time interval:   {TIME_INTERVAL[0]} to {TIME_INTERVAL[1]}")
    print(f"  Cloud threshold: < {MAX_CLOUD_COVER}%")
    print(f"  Target tile:     {TARGET_TILE}")
    print(f"  Scenes to select: {N_SCENES}")

    # Load credentials and build config
    try:
        client_id, client_secret = load_credentials(CREDENTIALS_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"FAILED to load credentials: {e}")
        return 1
    config = build_config(client_id, client_secret)

    # Query catalog
    print("\nQuerying STAC catalog...")
    try:
        all_scenes = query_catalog(config)
    except Exception as e:
        print(f"FAILED query: {type(e).__name__}: {e}")
        return 1
    print(f"  Retrieved {len(all_scenes)} total scenes.")

    # Filter to target tile
    tile_scenes = filter_to_tile(all_scenes, TARGET_TILE)
    print(f"  Filtered to tile {TARGET_TILE}: {len(tile_scenes)} scenes.")

    if not tile_scenes:
        print(f"\nNo scenes found for tile {TARGET_TILE}. Check tile identifier.")
        return 1

    # Select N scenes from time buckets
    print(f"\nSelecting {N_SCENES} scenes via time-bucket lowest-cloud strategy...")
    selected = select_by_time_buckets(tile_scenes, N_SCENES)

    # Print and persist
    print_summary(selected, len(tile_scenes), len(all_scenes))

    print(f"Writing manifest to: {MANIFEST_PATH}")
    write_manifest(selected, MANIFEST_PATH)
    print(f"Manifest written ({len(selected)} rows).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

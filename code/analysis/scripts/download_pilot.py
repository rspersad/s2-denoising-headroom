"""
Stage 4: Download Sentinel-2 L2A band files via Copernicus Data Space OData API.

Purpose:
    For each scene in the pilot selection manifest, download the band files
    required by the Praxis (visible: B02, B03, B04; red-edge: B05, B06, B07;
    SWIR: B11, B12).

    Uses the Copernicus Data Space OData v1 API for authenticated access to
    the original .jp2 files in the ESA archive.

Authentication note:
    CDSE has two OAuth flows with different audience claims on issued tokens:

    1. client_credentials with Sentinel Hub OAuth client (sh-*) — produces
       tokens valid for STAC catalog queries and Sentinel Hub APIs. Used by
       auth_test.py, catalog_query.py, and select_scenes.py.

    2. password flow with cdse-public client — produces tokens with audience
       for the download service. Required for file downloads via OData
       $value endpoints. Used by THIS script.

Redirect handling:
    The CDSE OData $value endpoint on catalogue.dataspace.copernicus.eu
    returns a 301 redirect to download.dataspace.copernicus.eu for the
    actual file content. The requests library (and curl with plain -L)
    strips Authorization headers on cross-host redirects as a security
    measure. This script handles redirects manually so that the
    Authorization header is explicitly re-applied to each hop. This is
    equivalent to curl's --location-trusted option.

Methodology:
    1. Authenticate via OAuth2 password flow with cdse-public client.
    2. Read selection manifest produced by select_scenes.py.
    3. For each scene, look up the product UUID via OData catalog.
    4. Navigate the product's nested .SAFE directory structure.
    5. For each band file: GET the $value URL with manual redirect handling.
    6. Skip files already present with correct size.
    7. Write a download manifest CSV.

Usage:
    python download_pilot.py

Author: Rajesh Persad
"""

from pathlib import Path
import sys
import csv
import time
from datetime import datetime
from typing import Optional

import requests


# Paths
CREDENTIALS_PATH = Path.home() / ".copernicus_credentials"
SELECTION_MANIFEST = Path("/data/praxis/manifests/sentinel2_pilot_selection.csv")
DOWNLOAD_MANIFEST = Path("/data/praxis/manifests/sentinel2_pilot_downloads.csv")
RAW_DATA_ROOT = Path("/data/praxis/raw/sentinel2")

# CDSE endpoints
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"

# Public OAuth client for download flow.
CDSE_PUBLIC_CLIENT_ID = "cdse-public"

# Bands to download, mapped to spectral category and resolution.
BAND_CATEGORIES = {
    "B02": ("visible", "R10m"),   # Blue, 490nm
    "B03": ("visible", "R10m"),   # Green, 560nm
    "B04": ("visible", "R10m"),   # Red, 665nm
    "B05": ("red-edge", "R20m"),  # 705nm
    "B06": ("red-edge", "R20m"),  # 740nm
    "B07": ("red-edge", "R20m"),  # 783nm
    "B11": ("swir", "R20m"),      # 1610nm
    "B12": ("swir", "R20m"),      # 2190nm
}

# Network behavior
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
REQUEST_TIMEOUT = 120
MAX_RETRIES_PER_FILE = 3
RETRY_BACKOFF_SECONDS = 5
MAX_REDIRECTS = 5

# Refresh token before this many seconds elapsed since issue.
# CDSE tokens are valid for 600s; we refresh well before to handle long files.
TOKEN_LIFETIME_BUFFER = 540


# ---------- Credentials and authentication ----------

def load_credentials(path: Path) -> dict[str, str]:
    """Read KEY=VALUE credentials from a file, validating required keys."""
    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found at {path}")
    creds: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        creds[key.strip()] = value.strip()
    missing = [k for k in ("COPERNICUS_USERNAME", "COPERNICUS_PASSWORD") if not creds.get(k)]
    if missing:
        raise ValueError(
            f"Credentials file is missing required keys: {missing}. "
            "Add COPERNICUS_USERNAME=<email> and COPERNICUS_PASSWORD=<password>."
        )
    return creds


class TokenManager:
    """
    Manages an OAuth access token via the CDSE password flow.

    Token is fetched lazily and refreshed when older than TOKEN_LIFETIME_BUFFER
    seconds. Caller reads .token property which transparently handles refresh.
    """
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._access_token: Optional[str] = None
        self._issued_at: float = 0.0

    def _fetch(self) -> str:
        response = requests.post(
            CDSE_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": CDSE_PUBLIC_CLIENT_ID,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if "access_token" not in payload:
            raise RuntimeError(f"Token response missing access_token: {payload}")
        return payload["access_token"]

    @property
    def token(self) -> str:
        now = time.time()
        if self._access_token is None or (now - self._issued_at) > TOKEN_LIFETIME_BUFFER:
            self._access_token = self._fetch()
            self._issued_at = now
        return self._access_token


# ---------- Product lookup ----------

def find_product_by_name(scene_id: str, token: str) -> Optional[dict]:
    """Look up product metadata by scene name via the OData catalog."""
    url = f"{CDSE_ODATA_BASE}/Products"
    params = {"$filter": f"Name eq '{scene_id}'", "$top": 1}
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    items = response.json().get("value", [])
    return items[0] if items else None


# ---------- Path-based node navigation ----------

def list_node_children(node_url: str, token: str) -> list[dict]:
    """List the children of a node by GETting its URL."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(node_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json().get("result", [])


def build_path_url(product_id: str, path_segments: list[str]) -> str:
    """Build a CDSE OData path-based URL using nested Nodes() segments."""
    url = f"{CDSE_ODATA_BASE}/Products({product_id})"
    for segment in path_segments:
        url += f"/Nodes({segment})"
    return url


def find_band_files(product_id: str, scene_id: str, token: str) -> dict[str, dict]:
    """
    Navigate the product's .SAFE directory to enumerate band files.

    Returns a dict mapping band code (e.g., "B02") to:
        - "name": filename
        - "path_segments": list of node names from .SAFE root to the file
        - "size": file size in bytes (if available)
    """
    band_files: dict[str, dict] = {}

    # Top-level: find .SAFE root
    product_nodes_url = f"{CDSE_ODATA_BASE}/Products({product_id})/Nodes"
    top_children = list_node_children(product_nodes_url, token)
    safe_node = next(
        (n for n in top_children if n["Name"] == scene_id or n["Name"].endswith(".SAFE")),
        None,
    )
    if not safe_node:
        raise RuntimeError(f"Could not find .SAFE root for {scene_id}")
    safe_name = safe_node["Name"]

    # .SAFE → GRANULE
    safe_children = list_node_children(
        build_path_url(product_id, [safe_name]) + "/Nodes", token
    )
    granule_node = next((n for n in safe_children if n["Name"] == "GRANULE"), None)
    if not granule_node:
        raise RuntimeError(f"No GRANULE folder in {scene_id}")

    # GRANULE → L2A_T<tile>_*
    granules = list_node_children(
        build_path_url(product_id, [safe_name, "GRANULE"]) + "/Nodes", token
    )
    if not granules:
        raise RuntimeError(f"GRANULE folder is empty for {scene_id}")
    granule_subdir = granules[0]["Name"]

    # IMG_DATA → R10m, R20m
    resolution_dirs = list_node_children(
        build_path_url(product_id, [safe_name, "GRANULE", granule_subdir, "IMG_DATA"]) + "/Nodes",
        token,
    )
    for res_dir in resolution_dirs:
        res_name = res_dir["Name"]
        if res_name not in {"R10m", "R20m"}:
            continue
        files = list_node_children(
            build_path_url(
                product_id, [safe_name, "GRANULE", granule_subdir, "IMG_DATA", res_name]
            ) + "/Nodes",
            token,
        )
        for f in files:
            fname = f["Name"]
            if not fname.endswith(".jp2"):
                continue
            for band_code, (_, expected_res) in BAND_CATEGORIES.items():
                if f"_{band_code}_" in fname and res_name == expected_res:
                    band_files[band_code] = {
                        "name": fname,
                        "path_segments": [
                            safe_name, "GRANULE", granule_subdir,
                            "IMG_DATA", res_name, fname,
                        ],
                        "size": f.get("ContentLength", 0),
                    }
                    break

    return band_files


# ---------- Downloading with manual redirect handling ----------

def download_file_with_manual_redirects(
    url: str, dest: Path, expected_size: int, token_mgr: "TokenManager"
) -> tuple[bool, str]:
    """
    Download a file with manual redirect handling and explicit auth re-application.

    The CDSE catalogue host returns a 301 to download.dataspace.copernicus.eu;
    we manually follow each redirect and re-send the Authorization header.
    This is the Python equivalent of curl --location-trusted.

    Refreshes the token if it's about to expire (via TokenManager).
    """
    # Skip if already present and correct size
    if dest.exists() and expected_size > 0 and dest.stat().st_size == expected_size:
        return True, "skipped"
    if dest.exists() and expected_size == 0 and dest.stat().st_size > 0:
        return True, "skipped (no size to verify)"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(1, MAX_RETRIES_PER_FILE + 1):
        try:
            current_url = url
            redirect_count = 0
            final_response = None

            while redirect_count < MAX_REDIRECTS:
                # Get current token (refreshes automatically if near expiry)
                headers = {"Authorization": f"Bearer {token_mgr.token}"}
                response = requests.get(
                    current_url,
                    headers=headers,
                    allow_redirects=False,
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code in (301, 302, 303, 307, 308):
                    new_url = response.headers.get("Location")
                    if not new_url:
                        response.close()
                        return False, f"Redirect (status {response.status_code}) with no Location"
                    current_url = new_url
                    redirect_count += 1
                    response.close()
                    continue

                # Not a redirect; this is the final response
                final_response = response
                break
            else:
                return False, f"Exceeded max redirects ({MAX_REDIRECTS})"

            if final_response is None:
                return False, "No final response after redirects"

            final_response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in final_response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            final_response.close()

            # Atomic move to final destination
            tmp_path.replace(dest)
            return True, "downloaded"

        except (requests.exceptions.RequestException, OSError) as e:
            if attempt < MAX_RETRIES_PER_FILE:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            else:
                if tmp_path.exists():
                    tmp_path.unlink()
                return False, f"{type(e).__name__}: {e}"

    return False, "exhausted retries"


def filename_for_band(scene_id: str, band_code: str, original_name: str) -> str:
    """Construct a deterministic filename including scene identifier and band."""
    base = scene_id.replace(".SAFE", "")
    if original_name.endswith("_10m.jp2"):
        return f"{base}_{band_code}_10m.jp2"
    elif original_name.endswith("_20m.jp2"):
        return f"{base}_{band_code}_20m.jp2"
    else:
        return f"{base}_{band_code}.jp2"


# ---------- Manifest handling ----------

def read_selection_manifest(path: Path) -> list[dict]:
    """Read the selection manifest produced by select_scenes.py."""
    if not path.exists():
        raise FileNotFoundError(f"Selection manifest not found at {path}")
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_download_record(path: Path, record: dict, write_header: bool) -> None:
    """Append a record to the download manifest CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id", "band", "filepath", "size_bytes",
        "status", "message", "download_timestamp",
    ]
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header and not file_exists:
            writer.writeheader()
        writer.writerow(record)


# ---------- Main orchestration ----------

def main() -> int:
    print("Sentinel-2 pilot download via Copernicus OData API (v4)")
    print("=" * 100)

    try:
        creds = load_credentials(CREDENTIALS_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"FAILED: {e}")
        return 1

    print(f"Username: {creds['COPERNICUS_USERNAME']}")
    print(f"OAuth client: {CDSE_PUBLIC_CLIENT_ID}")
    print()

    token_mgr = TokenManager(creds["COPERNICUS_USERNAME"], creds["COPERNICUS_PASSWORD"])

    print("Verifying token retrieval...")
    try:
        _ = token_mgr.token
    except (requests.exceptions.RequestException, RuntimeError) as e:
        print(f"FAILED authentication: {e}")
        return 1
    print("  OK\n")

    print(f"Reading selection manifest: {SELECTION_MANIFEST}")
    try:
        scenes = read_selection_manifest(SELECTION_MANIFEST)
    except FileNotFoundError as e:
        print(f"FAILED: {e}")
        return 1
    print(f"  {len(scenes)} scenes to process\n")

    total_files_attempted = 0
    total_files_downloaded = 0
    total_files_skipped = 0
    total_files_failed = 0
    total_bytes = 0
    start_time = time.time()
    header_needed = not DOWNLOAD_MANIFEST.exists()

    for scene_idx, scene_row in enumerate(scenes, start=1):
        scene_id = scene_row["scene_id"]
        print(f"[{scene_idx}/{len(scenes)}] {scene_id}")

        # Look up the product
        try:
            product = find_product_by_name(scene_id, token_mgr.token)
        except requests.exceptions.RequestException as e:
            print(f"  Product lookup failed: {e}")
            continue
        if not product:
            print(f"  Product not found in catalog. Skipping.")
            continue

        product_id = product["Id"]

        # Enumerate band files
        try:
            band_files = find_band_files(product_id, scene_id, token_mgr.token)
        except (requests.exceptions.RequestException, RuntimeError) as e:
            print(f"  Failed to enumerate band files: {e}")
            continue

        missing = sorted(set(BAND_CATEGORIES) - set(band_files))
        if missing:
            print(f"  Warning: missing band entries: {missing}")

        for band_code in BAND_CATEGORIES:
            total_files_attempted += 1
            ts = datetime.now().isoformat()

            if band_code not in band_files:
                print(f"  {band_code}: NOT FOUND in product")
                total_files_failed += 1
                append_download_record(DOWNLOAD_MANIFEST, {
                    "scene_id": scene_id, "band": band_code,
                    "filepath": "", "size_bytes": 0,
                    "status": "not_found",
                    "message": "Band file not located in product",
                    "download_timestamp": ts,
                }, write_header=header_needed)
                header_needed = False
                continue

            band_info = band_files[band_code]
            category, _ = BAND_CATEGORIES[band_code]
            out_filename = filename_for_band(scene_id, band_code, band_info["name"])
            out_path = RAW_DATA_ROOT / category / out_filename

            download_url = build_path_url(product_id, band_info["path_segments"]) + "/$value"

            success, msg = download_file_with_manual_redirects(
                download_url, out_path, band_info["size"], token_mgr
            )

            if success and "skipped" in msg:
                total_files_skipped += 1
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  {band_code}: skipped (exists, {size_mb:.1f} MB)")
                total_bytes += out_path.stat().st_size
                status = "skipped"
            elif success:
                total_files_downloaded += 1
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  {band_code}: downloaded ({size_mb:.1f} MB) -> {category}/")
                total_bytes += out_path.stat().st_size
                status = "downloaded"
            else:
                total_files_failed += 1
                print(f"  {band_code}: FAILED - {msg}")
                status = "failed"

            append_download_record(DOWNLOAD_MANIFEST, {
                "scene_id": scene_id, "band": band_code,
                "filepath": str(out_path) if success else "",
                "size_bytes": out_path.stat().st_size if (success and out_path.exists()) else 0,
                "status": status,
                "message": msg,
                "download_timestamp": ts,
            }, write_header=header_needed)
            header_needed = False

    elapsed = time.time() - start_time
    total_gb = total_bytes / (1024 ** 3)

    print()
    print("=" * 100)
    print(f"Download complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Files attempted:  {total_files_attempted}")
    print(f"  Files downloaded: {total_files_downloaded}")
    print(f"  Files skipped:    {total_files_skipped}")
    print(f"  Files failed:     {total_files_failed}")
    print(f"  Total data:       {total_gb:.2f} GB")
    print(f"  Manifest:         {DOWNLOAD_MANIFEST}")

    return 0 if total_files_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

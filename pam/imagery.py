"""Image loading and imagery-provider adapters (FR-01, FR-11).

Uploads: GeoTIFF (GSD + CRS read from georeferencing) or PNG/JPEG + explicit GSD.
Address flow: geocode -> parcel -> fetch tile from a configured provider. Providers
need credentials supplied through environment variables / AWS Secrets Manager.
"""
from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .config import MAX_GSD, MIN_GSD

Image.MAX_IMAGE_PIXELS = 120_000_000


class ImageryError(ValueError):
    pass


@dataclass
class LoadedImage:
    pixels: np.ndarray            # HxWx3 uint8
    gsd: float                    # metres / pixel
    transform: Optional[object] = None  # rasterio Affine when georeferenced
    crs: Optional[str] = None
    source: str = "upload"


def _metres_per_unit(crs) -> Optional[float]:
    if crs is None:
        return None
    if crs.is_projected:
        return float(crs.linear_units_factor[1]) if crs.linear_units_factor else 1.0
    return None


def load_image(data: bytes, filename: str = "", gsd: float | None = None) -> LoadedImage:
    name = filename.lower()
    if name.endswith((".tif", ".tiff")) or data[:4] in (b"II*\x00", b"MM\x00*"):
        try:
            import rasterio
            with rasterio.MemoryFile(data) as mf, mf.open() as ds:
                arr = ds.read(indexes=[1, 2, 3] if ds.count >= 3 else [1, 1, 1])
                arr = np.moveaxis(arr, 0, -1)
                if arr.dtype != np.uint8:
                    lo, hi = np.percentile(arr, (1, 99))
                    arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                tr, crs = ds.transform, ds.crs
                if gsd is None and crs is not None:
                    mpu = _metres_per_unit(crs)
                    if mpu:
                        gsd = abs(tr.a) * mpu
                    elif crs.is_geographic:  # degrees -> metres at tile centre latitude
                        lat = (ds.bounds.top + ds.bounds.bottom) / 2
                        gsd = abs(tr.a) * 111_320 * math.cos(math.radians(lat))
                img = LoadedImage(arr, gsd or 0.0, tr if crs is not None else None,
                                  crs.to_string() if crs else None, "geotiff")
        except ImportError as e:  # pragma: no cover
            raise ImageryError("GeoTIFF support requires rasterio") from e
    else:
        try:
            arr = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:
            raise ImageryError(f"cannot read image: {e}") from e
        img = LoadedImage(arr, gsd or 0.0, source="raster")
    if not img.gsd:
        raise ImageryError("ground sample distance unknown: upload a GeoTIFF or pass gsd (metres per pixel)")
    if not (MIN_GSD <= img.gsd <= MAX_GSD):
        raise ImageryError(f"gsd {img.gsd:.3f} m/px outside supported range {MIN_GSD}-{MAX_GSD}")
    h, w = img.pixels.shape[:2]
    if h < 32 or w < 32:
        raise ImageryError("image too small")
    return img


# --------------------------------------------------------------------------- providers
def web_mercator_gsd(lat: float, zoom: int, scale: int = 1) -> float:
    """Metres per pixel of a Web-Mercator map at a latitude (Google/Mapbox static images)."""
    return 156_543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / scale


class ImageryProvider:
    name = "base"

    def available(self) -> bool:
        return False

    def fetch(self, lat: float, lon: float, size_m: float) -> LoadedImage:  # pragma: no cover
        raise NotImplementedError


class GoogleStaticProvider(ImageryProvider):
    """Google Maps Static API, satellite layer. Needs PAM_GOOGLE_MAPS_KEY.
    Check Google's terms for automated measurement use before production."""
    name = "google"

    def available(self):
        return bool(os.environ.get("PAM_GOOGLE_MAPS_KEY"))

    def fetch(self, lat, lon, size_m=60.0):
        import httpx
        zoom, scale = 20, 2
        gsd = web_mercator_gsd(lat, zoom, scale)
        px = int(min(640, max(200, size_m / gsd / scale)))
        r = httpx.get("https://maps.googleapis.com/maps/api/staticmap", timeout=20, params={
            "center": f"{lat},{lon}", "zoom": zoom, "scale": scale, "size": f"{px}x{px}",
            "maptype": "satellite", "key": os.environ["PAM_GOOGLE_MAPS_KEY"]})
        r.raise_for_status()
        return LoadedImage(np.array(Image.open(io.BytesIO(r.content)).convert("RGB")), gsd, source="google")


class MapboxProvider(ImageryProvider):
    """Mapbox Static Images API, satellite style. Needs PAM_MAPBOX_TOKEN."""
    name = "mapbox"

    def available(self):
        return bool(os.environ.get("PAM_MAPBOX_TOKEN"))

    def fetch(self, lat, lon, size_m=60.0):
        import httpx
        zoom = 19
        gsd = web_mercator_gsd(lat, zoom, 2)
        px = int(min(1280, max(256, size_m / gsd / 2)))
        url = (f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/"
               f"{px}x{px}@2x")
        r = httpx.get(url, params={"access_token": os.environ["PAM_MAPBOX_TOKEN"]}, timeout=20)
        r.raise_for_status()
        return LoadedImage(np.array(Image.open(io.BytesIO(r.content)).convert("RGB")), gsd, source="mapbox")


class NearmapProvider(ImageryProvider):
    """Nearmap tile API (0.075 m class imagery). Needs PAM_NEARMAP_KEY. Tile stitching
    to be completed against the client's Nearmap contract (endpoint varies by plan)."""
    name = "nearmap"

    def available(self):
        return bool(os.environ.get("PAM_NEARMAP_KEY"))

    def fetch(self, lat, lon, size_m=60.0):
        raise ImageryError("Nearmap adapter requires the client's contract endpoint; see README")


PROVIDERS = {p.name: p for p in (GoogleStaticProvider(), MapboxProvider(), NearmapProvider())}


def geocode(address: str) -> tuple[float, float]:
    """Geocode with the US Census geocoder (free, no key). Override in production."""
    import httpx
    r = httpx.get("https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
                  params={"address": address, "benchmark": "Public_AR_Current", "format": "json"},
                  timeout=20)
    r.raise_for_status()
    matches = r.json().get("result", {}).get("addressMatches", [])
    if not matches:
        raise ImageryError(f"address not found: {address}")
    c = matches[0]["coordinates"]
    return float(c["y"]), float(c["x"])

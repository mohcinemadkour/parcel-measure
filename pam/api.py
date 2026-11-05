"""REST API for the Property Area Measurement prototype (spec section 8).

Run locally:  uvicorn pam.api:app --host 0.0.0.0 --port 8080
Docs:         /docs (Swagger UI), /openapi.json
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .config import CLASS_COLORS, CLASSES, ROOT
from .imagery import PROVIDERS, ImageryError, geocode, load_image
from .measure import measure
from .quote import DEFAULT_PRICE_TABLE, PriceTable, build_quote
from .storage import make_store, put_json

log = logging.getLogger("pam")
logging.basicConfig(level=os.environ.get("PAM_LOG_LEVEL", "INFO"),
                    format='{"t":"%(asctime)s","lvl":"%(levelname)s","msg":%(message)s}')

WEB_DIR = Path(os.environ.get("PAM_WEB_DIR", ROOT / "web"))
SAMPLES_DIR = WEB_DIR / "samples"
MAX_UPLOAD_MB = float(os.environ.get("PAM_MAX_UPLOAD_MB", "40"))
RATE_PER_MIN = int(os.environ.get("PAM_RATE_LIMIT_PER_MIN", "120"))


def _api_keys() -> set[str]:
    raw = os.environ.get("PAM_API_KEYS", "")
    if raw.strip().startswith("["):
        return set(json.loads(raw))
    return {k.strip() for k in raw.split(",") if k.strip()}


API_KEYS = _api_keys()

@asynccontextmanager
async def lifespan(_app):
    if not API_KEYS:
        log.warning('"PAM_API_KEYS not set: API key check disabled (development mode)"')
    try:
        get_segmenter()
    except HTTPException:
        pass
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Property Area Measurement API",
    version=__version__,
    description="Measure lawn, roof, hardscape and tree-canopy areas from overhead imagery and price a quote.",
)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("PAM_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

_state: dict = {"segmenter": None, "store": None, "error": None}
_buckets: dict[str, list[float]] = defaultdict(list)
_rl_lock = threading.Lock()


def get_segmenter():
    if _state["segmenter"] is None and _state["error"] is None:
        try:
            from .inference import Segmenter
            _state["segmenter"] = Segmenter(threads=int(os.environ.get("PAM_THREADS", "0")) or None)
        except Exception as e:  # surfaced through /v1/ready
            _state["error"] = str(e)
    if _state["segmenter"] is None:
        raise HTTPException(503, f"model not loaded: {_state['error']}")
    return _state["segmenter"]


def get_store():
    if _state["store"] is None:
        _state["store"] = make_store()
    return _state["store"]


def require_key(request: Request, x_api_key: Optional[str] = Header(default=None)) -> str:
    key = x_api_key or request.query_params.get("api_key") or ""
    if API_KEYS and key not in API_KEYS:
        raise HTTPException(401, "missing or invalid X-API-Key")
    ident = hashlib.sha256(key.encode()).hexdigest()[:12] if key else (request.client.host if request.client else "anon")
    now = time.time()
    with _rl_lock:
        q = [t for t in _buckets[ident] if now - t < 60]
        if len(q) >= RATE_PER_MIN:
            raise HTTPException(429, "rate limit exceeded")
        q.append(now)
        _buckets[ident] = q
    return ident


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t = time.time()
    resp = await call_next(request)
    resp.headers["x-request-id"] = rid
    if request.url.path.startswith("/v1/"):
        log.info(json.dumps({"rid": rid, "path": request.url.path, "method": request.method,
                             "status": resp.status_code, "ms": round((time.time() - t) * 1000, 1)}))
    return resp


# ------------------------------------------------------------------ helpers
def overlay_png(image: np.ndarray, classes: np.ndarray, valid: np.ndarray, max_side: int = 1600) -> bytes:
    pal = np.array([CLASS_COLORS[c] for c in CLASSES], np.float32)
    rgba = pal[classes]
    rgba[~valid, 3] = 0
    a = rgba[..., 3:4] / 255.0
    base = image.astype(np.float32)
    base[~valid] *= 0.45  # dim outside the parcel
    out = (base * (1 - a) + rgba[..., :3] * a).astype(np.uint8)
    im = Image.fromarray(out)
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((round(im.size[0] * f), round(im.size[1] * f)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _parse_json(s: Optional[str], name: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"{name} is not valid JSON: {e}")


def _run_measurement(data: bytes, filename: str, gsd: Optional[float], parcel, pitch: Optional[float],
                     table: PriceTable, include_overlay: bool, source: str = "upload") -> dict:
    t0 = time.time()
    try:
        img = load_image(data, filename, gsd)
    except ImageryError as e:
        raise HTTPException(422, str(e))
    if isinstance(parcel, dict):  # GeoJSON Polygon in image CRS -> pixel coords
        coords = parcel.get("coordinates", [[]])[0]
        if img.transform is None:
            raise HTTPException(422, "GeoJSON parcel needs a georeferenced GeoTIFF; send pixel coordinates instead")
        inv = ~img.transform
        parcel = [list(inv * (x, y)) for x, y in coords]
    seg = get_segmenter()
    res = seg.predict(img.pixels, img.gsd)
    m = measure(res.classes, res.probs, res.scale, img.gsd, parcel, pitch, img.transform)
    cls, valid = m.pop("_clean_classes"), m.pop("_valid")
    rid = uuid.uuid4().hex
    result = {
        "id": rid,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": seg.version,
        "api_version": __version__,
        "input": {"filename": filename, "source": source, "sha256": hashlib.sha256(data).hexdigest(),
                  "width": int(img.pixels.shape[1]), "height": int(img.pixels.shape[0]),
                  "gsd_m": round(img.gsd, 4), "crs": img.crs, "parcel": parcel, "pitch": pitch},
        **m,
        "quote": build_quote(m["measurements"], table),
        "price_table": table.model_dump(),
        "timing_ms": round((time.time() - t0) * 1000, 1),
    }
    png = overlay_png(res.image, cls, valid)
    store = get_store()
    put_json(store, f"results/{rid}.json", result)
    store.put(f"results/{rid}_overlay.png", png, "image/png")
    if os.environ.get("PAM_STORE_INPUTS", "false").lower() == "true":
        store.put(f"inputs/{rid}_{Path(filename).name or 'image'}", data)
    if include_overlay:
        result = {**result, "overlay_png_base64": base64.b64encode(png).decode()}
    return result


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def index():
    p = WEB_DIR / "index.html"
    return FileResponse(p) if p.exists() else HTMLResponse("<h1>PAM API</h1><a href='/docs'>docs</a>")


if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/v1/health", tags=["ops"])
def health():
    return {"status": "ok", "version": __version__}


@app.get("/v1/ready", tags=["ops"])
def ready():
    try:
        get_segmenter()
        return {"ready": True, "store": get_store().describe()}
    except HTTPException as e:
        return JSONResponse({"ready": False, "error": e.detail}, status_code=503)


@app.get("/v1/model", tags=["model"])
def model_card():
    return get_segmenter().card


@app.get("/v1/price-table/default", tags=["quote"])
def default_price_table():
    return DEFAULT_PRICE_TABLE.model_dump()


@app.get("/v1/samples", tags=["samples"])
def samples():
    p = SAMPLES_DIR / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


@app.post("/v1/measure", tags=["measure"])
async def measure_upload(
    file: UploadFile = File(..., description="GeoTIFF, PNG or JPEG"),
    gsd: Optional[float] = Form(None, description="metres per pixel; required unless GeoTIFF"),
    parcel: Optional[str] = Form(None, description="JSON: [[x,y],...] pixel polygon or GeoJSON Polygon in image CRS"),
    pitch: Optional[float] = Form(None, ge=0, le=24, description="roof pitch, rise per 12 run (default 6)"),
    price_table: Optional[str] = Form(None, description="JSON PriceTable; default table if omitted"),
    include_overlay: bool = Form(True),
    _: str = Depends(require_key),
):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1e6:
        raise HTTPException(413, f"upload larger than {MAX_UPLOAD_MB} MB")
    table = PriceTable(**_parse_json(price_table, "price_table")) if price_table else DEFAULT_PRICE_TABLE
    return _run_measurement(data, file.filename or "", gsd, _parse_json(parcel, "parcel"), pitch, table,
                            include_overlay)


@app.post("/v1/measure/batch", tags=["measure"])
async def measure_batch(files: list[UploadFile] = File(...), gsd: Optional[float] = Form(None),
                        pitch: Optional[float] = Form(None), _: str = Depends(require_key)):
    if len(files) > 50:
        raise HTTPException(413, "max 50 images per batch")
    out = []
    for f in files:
        try:
            r = _run_measurement(await f.read(), f.filename or "", gsd, None, pitch, DEFAULT_PRICE_TABLE, False)
            out.append({"filename": f.filename, "id": r["id"], "measurements": r["measurements"],
                        "review": r["review"], "quote_total": r["quote"]["total"]})
        except HTTPException as e:
            out.append({"filename": f.filename, "error": e.detail})
    return {"results": out}


class AddressRequest(BaseModel):
    address: str
    provider: str = Field("google", description="google | mapbox | nearmap")
    size_m: float = Field(60.0, ge=20, le=200, description="side of the square fetched around the point")
    pitch: Optional[float] = Field(None, ge=0, le=24)
    price_table: Optional[PriceTable] = None


@app.post("/v1/measure/address", tags=["measure"])
def measure_address(req: AddressRequest, _: str = Depends(require_key)):
    prov = PROVIDERS.get(req.provider)
    if not prov:
        raise HTTPException(422, f"unknown provider {req.provider}; options: {list(PROVIDERS)}")
    if not prov.available():
        raise HTTPException(503, f"imagery provider '{req.provider}' is not configured on this server "
                                 f"(set its API key in Secrets Manager / environment)")
    try:
        lat, lon = geocode(req.address)
        img = prov.fetch(lat, lon, req.size_m)
    except ImageryError as e:
        raise HTTPException(422, str(e))
    buf = io.BytesIO()
    Image.fromarray(img.pixels).save(buf, "PNG")
    r = _run_measurement(buf.getvalue(), f"{req.provider}.png", img.gsd, None, req.pitch,
                         req.price_table or DEFAULT_PRICE_TABLE, True, source=req.provider)
    r["input"]["address"], r["input"]["lat"], r["input"]["lon"] = req.address, lat, lon
    return r


class QuoteRequest(BaseModel):
    measurements: dict
    price_table: Optional[PriceTable] = None


@app.post("/v1/quote", tags=["quote"])
def quote(req: QuoteRequest, _: str = Depends(require_key)):
    """Re-price after a user edits a measurement or the price table."""
    return build_quote(req.measurements, req.price_table or DEFAULT_PRICE_TABLE)


def _load_result(rid: str) -> dict:
    if not rid.isalnum():
        raise HTTPException(400, "bad id")
    raw = get_store().get(f"results/{rid}.json")
    if raw is None:
        raise HTTPException(404, "result not found")
    return json.loads(raw)


@app.get("/v1/results/{rid}", tags=["results"])
def get_result(rid: str, _: str = Depends(require_key)):
    return _load_result(rid)


@app.get("/v1/results/{rid}/overlay.png", tags=["results"])
def get_overlay(rid: str, _: str = Depends(require_key)):
    _load_result(rid)
    return Response(get_store().get(f"results/{rid}_overlay.png"), media_type="image/png")


@app.get("/v1/results/{rid}/geojson", tags=["results"])
def get_geojson(rid: str, _: str = Depends(require_key)):
    return JSONResponse(_load_result(rid).get("geojson", {}), media_type="application/geo+json")


@app.get("/v1/results/{rid}/csv", tags=["results"])
def get_csv(rid: str, _: str = Depends(require_key)):
    r = _load_result(rid)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["result_id", "metric", "value"])
    for k, v in r["measurements"].items():
        w.writerow([rid, k, v])
    for it in r["quote"]["items"]:
        w.writerow([rid, f"quote:{it['service']}", it["price"]])
    w.writerow([rid, "quote:total", r["quote"]["total"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=pam_{rid}.csv"})


class Feedback(BaseModel):
    corrected: dict = Field(..., description="e.g. {'lawn_sqft': 5230}")
    reviewer: Optional[str] = None
    note: Optional[str] = None


@app.post("/v1/results/{rid}/feedback", tags=["results"])
def feedback(rid: str, fb: Feedback, _: str = Depends(require_key)):
    r = _load_result(rid)
    rec = {"result_id": rid, "model_version": r.get("model_version"), "predicted": r["measurements"],
           **fb.model_dump(), "received": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    put_json(get_store(), f"feedback/{rid}_{uuid.uuid4().hex[:6]}.json", rec)
    new = {**r["measurements"], **fb.corrected}
    return {"stored": True, "quote": build_quote(new, PriceTable(**r["price_table"]))}

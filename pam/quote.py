"""Quote engine: price table x measured quantities (spec section 7, FR-09)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Basis = Literal["lawn_sqft", "lawn_perimeter_ft", "roof_surface_sqft", "roof_squares",
                "roof_perimeter_ft", "hardscape_sqft", "tree_canopy_sqft", "flat"]


class PriceLine(BaseModel):
    service: str
    basis: Basis
    rate: float = Field(ge=0, description="price per unit")
    per: float = Field(1.0, gt=0, description="unit size, e.g. 1000 for 'per 1,000 sq ft'")
    minimum: float = Field(0.0, ge=0)
    surcharge_pct: float = Field(0.0, ge=0, le=500)
    enabled: bool = True


class PriceTable(BaseModel):
    company: str = "Demo Lawn & Home Services"
    currency: str = "USD"
    tax_pct: float = Field(0.0, ge=0, le=30)
    lines: list[PriceLine]


DEFAULT_PRICE_TABLE = PriceTable(lines=[
    PriceLine(service="Mowing (per visit)", basis="lawn_sqft", rate=4.50, per=1000, minimum=35),
    PriceLine(service="Fertilization (per application)", basis="lawn_sqft", rate=6.00, per=1000, minimum=45),
    PriceLine(service="Core aeration", basis="lawn_sqft", rate=9.00, per=1000, minimum=75),
    PriceLine(service="Edging", basis="lawn_perimeter_ft", rate=0.15, per=1, minimum=20),
    PriceLine(service="Perimeter pest control", basis="roof_perimeter_ft", rate=0.12, per=1, minimum=75),
    PriceLine(service="Roof replacement (asphalt shingle)", basis="roof_squares", rate=425.0, per=1, minimum=0,
              enabled=False),
    PriceLine(service="Driveway & patio pressure wash", basis="hardscape_sqft", rate=0.18, per=1, minimum=99,
              enabled=False),
])


def build_quote(measurements: dict, table: PriceTable = DEFAULT_PRICE_TABLE) -> dict:
    items, subtotal = [], 0.0
    for ln in table.lines:
        if not ln.enabled:
            continue
        qty = 1.0 if ln.basis == "flat" else float(measurements.get(ln.basis, 0.0) or 0.0)
        raw = qty / ln.per * ln.rate
        price = max(raw, ln.minimum) * (1 + ln.surcharge_pct / 100.0)
        items.append({
            "service": ln.service, "basis": ln.basis, "quantity": round(qty, 2),
            "unit": f"per {ln.per:g} {ln.basis.split('_')[-1]}" if ln.basis != "flat" else "flat",
            "rate": ln.rate, "minimum_applied": raw < ln.minimum, "price": round(price, 2),
        })
        subtotal += price
    tax = subtotal * table.tax_pct / 100.0
    return {"company": table.company, "currency": table.currency, "items": items,
            "subtotal": round(subtotal, 2), "tax": round(tax, 2), "total": round(subtotal + tax, 2)}

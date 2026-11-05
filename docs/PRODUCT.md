# parcel-measure: product, use cases and go-to-market

This document describes who the product is for, how customers use it, how to price and
sell it, and what else the engine can do. Technical details are in the main
[README](../README.md). Prices marked as "proposed" are suggestions to test with pilot
customers; competitor prices come from third-party pages and should be confirmed on each
vendor's pricing page before quoting them.

*Last updated: November 2026.*

## Product: use cases and who buys

The product is a **measurement engine**: give it a property, get back measured areas and a
price, in seconds, with no site visit. Anyone who prices work by the size of a yard, roof
or pavement is a buyer. Speed is what they pay for: Deep Lawn's CEO says
[80% of lawn and pest customers choose the company that gives the first estimate](https://www.serviceautopilot.com/?p=23132).

| Segment | Their problem today | What they measure with it | Value |
| --- | --- | --- | --- |
| Lawn care and landscaping companies | Drive out or trace yards by hand before quoting | Lawn area, lawn edge, beds | Instant quotes, more leads closed, consistent pricing |
| Pest control companies | Price perimeter treatments by guesswork | House outline length, lawn area | Quote by phone or web in one call |
| Roofing and gutter contractors | Pay per roof report and wait for it | Roof area (with pitch), roof outline | Lower cost per bid, same-day proposals |
| Snow removal, sealcoating, pressure washing | Measure parking lots and driveways manually | Paved (hardscape) area | Bid commercial lots in minutes |
| Field-service and quoting software vendors | Customers ask for built-in measurement | Everything, through the API | A paid feature inside their product |
| Property managers and HOAs | Budget landscaping across many sites | Portfolio totals by feature | Fair bids, checks on vendor invoices |
| Insurers and solar installers (later) | Need roof size and tree risk at scale | Roof area, tree canopy over roof | Faster underwriting and site surveys |

The U.S. lawn and landscape industry alone is about
[$176 billion a year](https://myquoteiq.com/?p=20348) (NALP figure as cited by QuoteIQ),
and most of it runs on recurring contracts priced by area.

## How customers use it

The same engine ships in five forms, all built on the API that is already running:

| Form | Who uses it | Status |
| --- | --- | --- |
| **Instant-quote widget** on the contractor's website | Homeowners type an address and buy a service | To build: small embeddable page on top of `/v1/measure/address` |
| **Estimator web app** (the current interface) | Sales reps check outlines, correct, send the quote | Built |
| **CRM integrations** (Jobber, Service Autopilot, LMN, Aspire, ServiceTitan) | Measurements land directly on the customer record | To build: marketplace apps calling the API |
| **Batch measurement** of a CSV of addresses | Commercial accounts, property managers, route planning | API ready (`/v1/measure/batch`); CSV upload page to add |
| **White-label API** | Quoting-software vendors embed it under their brand | Built (API keys, rate limits, exports) |

A typical day for a lawn care company using it:

1. At 9 pm a homeowner enters their address on the company's site.
2. The widget fetches imagery, measures 6,800 sq ft of lawn and shows mowing at $35 per
   visit, fertilization at $45 and an annual package.
3. The homeowner books online. The estimate and outline land in the company's CRM.
4. Next morning the office sees the booking. Any property flagged for review (heavy tree
   cover, low confidence) gets a 30-second check by a person.
5. The crew gets the mapped yard with the work order. Any correction made is saved and
   used to retrain the model.

## Business model and pricing

Recommended: **sell the engine to quoting-software vendors and larger contractors (API,
per measurement), funded at first by custom fine-tuning projects.** A self-serve
contractor app comes second, because that market is already crowded.

### What competitors charge today

| Product | What it does | Published price | Source |
| --- | --- | --- | --- |
| Deep Lawn | AI lawn measurement + instant-quote widget (no CRM) | $95/month with 40 searches to $500/month with 800 | [QuoteIQ comparison, May 2026](https://myquoteiq.com/?p=20348) |
| QuoteIQ MapMeasure Pro | Lawn, roof and paved-area AutoMeasure inside a CRM | From $74.99/month, 100 credits per measurement | [QuoteIQ](https://myquoteiq.com/top-10-ai-tools-for-lawn-care-businesses-in-2026/) |
| Attentive.ai Automeasure | Commercial site takeoffs, human-checked | $32 per 10-acre landscaping site; $37 with snow | [attentive.ai/pricing](https://attentive.ai/pricing) |
| EagleView | Roof reports with pitch | $32.75 small / $60 medium / $87 large roof (Bronze tier) | [yespress.io](https://yespress.io/eagleview) |
| Roofr | Roof reports | $19 per report on the free Starter plan | [Sagnik Bhattacharya, Oct 2026](https://sagnikbhattacharya.com/blog/roofers-ai-tools-measurements-calls-estimates) |
| Go iLawn | Manual measuring tools on aerial photos | From $19.95/month (older listing) | [Landscape Management](https://www.landscapemanagement.net/?p=70081) |

### Proposed price points (to test with pilots)

| Offer | Buyer | Proposed price | Reasoning |
| --- | --- | --- | --- |
| Wholesale API | Quoting-software vendors | $0.40–$1.00 per property, volume tiers, monthly minimum | They resell inside a $50–$500/month product; must leave them margin |
| Contractor plan | Lawn and pest companies | $79/month with 150 measurements, then $0.50 each | Undercuts Deep Lawn's entry tier on included volume |
| Roof area report | Roofers, gutter companies | $9–$12 per report | Below Roofr and EagleView; only after pitch accuracy is proven on US data |
| Commercial site takeoff | Snow, sealcoating, facility maintenance | $20–$30 per site | Below Attentive.ai; add human review for large sites |
| Custom model project | Vendors with their own data | Fixed-fee fine-tuning + evaluation report | Builds the US dataset and pays for GPU training |

### Unit economics

- **Compute is close to free.** One Fargate server measures a typical lot in about 1.2
  seconds, so roughly 3,000 lots per server-hour. At list Fargate prices (about $0.10 per
  hour for 2 vCPU / 4 GB, approximate) that is a few thousandths of a cent per measurement.
- **Fixed hosting** is about $100/month for one always-on server and the load balancer
  (estimate). Break-even is 2–3 contractor subscriptions.
- **Imagery is the real cost and the real risk.** The license must allow automated
  measurement and resale of derived data. Price the API only after the imagery contract
  is signed; it may add more per property than everything else combined.

## Go-to-market plan

Start where a buyer has already asked for this: quoting-software vendors looking to add
satellite measurement. Win a first custom project, use it to build the US dataset, then
sell the engine to the next vendors and to Dallas–Fort Worth contractors.

| Days | Goal | Actions | Done when |
| --- | --- | --- | --- |
| 0–30 | First paid project; fix lawn accuracy | Bid on vendor projects with the live demo and specification; secure imagery license; label 300+ DFW properties; GPU retrain | Lawn and roof ≥ 90% median accuracy on a US acceptance set |
| 31–60 | 3 pilot customers | Free 30-day pilot for 3 DFW lawn/pest companies: measure 50 of their properties and compare with their crews' numbers; build the quote widget | Written accuracy report per pilot; 2 agree to pay |
| 61–90 | Repeatable sales | Convert pilots; submit a Jobber marketplace app; pitch the API to 10 quoting/field-service software vendors | 2 paying contractors + 1 vendor in trial |

**Where to find buyers:** industry associations and their events reach owners who buy,
rather than other builders. In Texas that means the landscape and nursery association, the
state pest control association and roofing contractor groups; nationally, the Equip
Exposition and NALP events. Online: Jobber and Service Autopilot marketplaces, lawn-care
owner communities, and the vendors' own partner programs.

**Differentiators to lead with:**

1. **Published accuracy.** Every customer gets an evaluation report with measured accuracy
   on their own properties.
2. **Embeddable and white-label.** Vendors keep their brand; competitors like Deep Lawn
   sell their own widget.
3. **One engine for lawn, roof and pavement.** One integration covers several trades.
4. **Price.** Compute costs almost nothing, so the API can undercut per-report pricing.

## What else it can do

The engine already separates lawn, roof, pavement and trees. Each extension below reuses it
with a new output or a little more training.

| Extension | Built on | Buyer | Effort |
| --- | --- | --- | --- |
| Snow and sealcoating takeoffs | Hardscape area, parking lots | Snow, paving contractors | Low: pricing rules only |
| Roof report with pitch and facets | Roof mask + LiDAR or Solar API | Roofers, insurers | Medium |
| Solar pre-assessment | Roof area, orientation, tree shade | Solar installers | Medium |
| Tree risk over roofs and lines | Tree canopy + roof overlap | Insurers, utilities, tree services | Medium |
| Impervious-surface measurement | Hardscape + roof area | Cities and utilities that bill stormwater fees by paved area | Low to medium |
| Change detection over time | Two image dates compared | Tax assessors, HOAs (new pools, additions), insurers | Medium |
| Portfolio budgeting | Batch measurement | Property managers, HOAs, REITs | Low |
| Irrigation and fertilizer planning | Lawn zones and edges | Irrigation designers, lawn programs | Low |
| Mobile capture for missing imagery | Phone or drone photo + same model | Contractors in rural areas | Medium to high |

Best next bets: **snow/sealcoating** (no new model, reuses hardscape) and
**impervious-surface measurement** for municipalities (public-sector buyers who value a
documented accuracy report).

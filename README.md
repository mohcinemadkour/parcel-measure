# parcel-measure

**Property area measurement prototype.** 
Measures **lawn, roof, hardscape and tree-canopy** areas of a property from an overhead
image, then prices a quote. Ships as a trained model, a REST API with a built-in client
interface, a no-server in-browser demo, tests and an AWS deployment kit.

The full specification (requirements, data, model, API, AWS architecture, acceptance
criteria and measured results) is in the companion document *Property Area Measurement
Prototype: Specification*. `models/evaluation.json` holds the measured accuracy.

## Repository layout

| Path | What it is |
| --- | --- |
| `pam/` | Python package: inference (`inference.py`), measurement (`measure.py`), quoting (`quote.py`), image loading and imagery providers (`imagery.py`), storage (`storage.py`), REST API (`api.py`), evaluation (`evaluation.py`) |
| `web/index.html` | Client interface (served at `/` by the API; also the in-browser demo) |
| `web/samples/` | Held-out sample images with ground-truth label masks |
| `models/` | `pam_unet_int8w.onnx` (24.6 MB, used by the API and the browser demo), `model_card.json`, `evaluation.json`, per-property test results, training history. The full-precision `pam_unet.onnx` (97.7 MB) is not in git; `scripts/export_onnx.py` recreates it and the API prefers it when present |
| `scripts/` | `prepare_potsdam.py`, `train.py`, `export_onnx.py`, `evaluate.py`, `make_samples.py` |
| `tests/` | Measurement-math and API tests (`pytest`) |
| `infra/` | `cloudformation.yaml`, `deploy.sh`, `sagemaker_train.py` |
| `Dockerfile` | Serving image (FastAPI + ONNX Runtime, no PyTorch) |

## Run locally

```bash
pip install -r requirements-serve.txt
uvicorn pam.api:app --port 8080            # open http://localhost:8080  (API docs: /docs)
# with API keys:
PAM_API_KEYS="dev-key-123456789" uvicorn pam.api:app --port 8080
```

Or with Docker:

```bash
docker build -t pam-api .
docker run -p 8080:8080 -e PAM_API_KEYS=dev-key-123456789 pam-api
```

Call the API:

```bash
curl -X POST http://localhost:8080/v1/measure -H "X-API-Key: dev-key-123456789" \
  -F file=@web/samples/sample_01.png -F gsd=0.15 -F pitch=6
```

## Deploy to AWS

```bash
export PAM_API_KEYS="$(openssl rand -hex 24)"   # save this value; clients send it as X-API-Key
./infra/deploy.sh pam-prototype us-east-1
```

The script creates an ECR repository, builds and pushes the image, deploys
`infra/cloudformation.yaml` (ECS Fargate behind an ALB, encrypted S3 results bucket with
30-day expiry, Secrets Manager for API and imagery keys, CloudWatch logs and alarms, CPU
autoscaling) into the default VPC, and waits until `/v1/ready` answers. Set `CERT_ARN`
for HTTPS, `ALLOWED_CIDR` to restrict access, `DESIRED_COUNT=0` to park it at no compute cost.

To enable address lookup (`POST /v1/measure/address`), put provider keys in the
`<stack>/imagery-keys` secret and redeploy. Check each provider's terms for automated
measurement before production use.

Remove everything: `aws cloudformation delete-stack --stack-name pam-prototype`
(the results bucket is retained on purpose; empty and delete it separately).

## Reproduce training and evaluation

```bash
pip install -r requirements-train.txt
python scripts/prepare_potsdam.py --out data/tiles10    # ~540 MB, ISPRS Potsdam mirror on GitHub
# ImageNet ResNet-34 weights -> models/resnet34-43635321.pth (timm v0.1-weights release)
python scripts/train.py --epochs 12 --samples 1024 --crop 320
python scripts/export_onnx.py --ckpt models/run/best.pt --out models
python scripts/evaluate.py                              # writes models/evaluation.json
python scripts/make_samples.py                          # refreshes web/samples
pytest -q
```

On a GPU, use more epochs and larger crops (see `infra/sagemaker_train.py`).

## Accuracy definition

For each property and feature: `accuracy = 1 - |measured - true| / true`, scored only when
the true area is at least 20 m² (215 sq ft). Properties are 36 m × 36 m windows cut from
held-out tiles. The acceptance bar is a median of at least 90% for lawn and roof.

## Results (held-out test, 512 property windows)

| Feature | Median accuracy | Within ±10% | Total-area accuracy |
| --- | --- | --- | --- |
| Roof | 96.8% | 81.4% | 99.6% |
| Lawn | 85.4% | 38.8% | 99.5% |

Roof meets the 90% target; lawn does not yet (winter imagery makes grass and bare trees hard to separate). The int8-weight model in this repo scores within 0.1 point of these numbers. Details: `models/evaluation.json` and the specification document.

## Important limits

* **Data license:** the model is trained on the ISPRS Potsdam benchmark, which is licensed for research. Treat this repository as a prototype and portfolio piece; retrain on licensed data before any commercial use.
* The model is trained on the ISPRS Potsdam benchmark (German city, leaf-off season,
  research license). Before production, fine-tune and re-validate on licensed US imagery
  with the calibration set described in the specification (section 5, AC-11).
* Roof surface area uses the pitch you supply (default 6/12). Plug in LiDAR or the Google
  Solar API for per-roof pitch.
* Lawn hidden under tree canopy is counted as tree canopy and flagged for review.

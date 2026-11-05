"""Launch a GPU retraining job on Amazon SageMaker (spec section 6, "Retraining").

Expects training tiles in S3 in the same layout as data/tiles10 (<tile>_img.png,
<tile>_lbl.png at 0.10 m) and a splits file. US labels from the calibration set and
feedback corrections are converted to that layout by the labeling pipeline.

Usage:
  python infra/sagemaker_train.py --bucket <bucket> --role-arn <SageMaker execution role> \
      [--instance ml.g5.xlarge] [--epochs 40]

The job runs scripts/train.py inside the AWS PyTorch training container, then
scripts/export_onnx.py; artifacts land in s3://<bucket>/pam/training/<job>/output/.
"""
import argparse
import io
import tarfile
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
ENTRY = """#!/bin/bash
set -e
cd /opt/ml/input/data/code && tar xzf code.tar.gz
pip install -q segmentation-models-pytorch timm onnx onnxscript onnxruntime scipy rasterio shapely
python scripts/train.py --data /opt/ml/input/data/tiles --out /opt/ml/model/run --epochs ${EPOCHS} \
  --samples 4096 --crop 512 --batch 16 --threads 8 --weights /opt/ml/input/data/code/models/resnet34-43635321.pth
python scripts/export_onnx.py --ckpt /opt/ml/model/run/best.pt --out /opt/ml/model
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--instance", default="ml.g5.xlarge")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tiles-prefix", default="pam/training/tiles10/")
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name=a.region)
    sm = boto3.client("sagemaker", region_name=a.region)
    job = f"pam-train-{time.strftime('%Y%m%d-%H%M%S')}"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in ["pam", "scripts", "requirements-train.txt", "models/resnet34-43635321.pth"]:
            tar.add(ROOT / p, arcname=p)
    s3.put_object(Bucket=a.bucket, Key=f"pam/training/{job}/code/code.tar.gz", Body=buf.getvalue())
    s3.put_object(Bucket=a.bucket, Key=f"pam/training/{job}/code/entry.sh", Body=ENTRY.encode())

    image = f"763104351884.dkr.ecr.{a.region}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker"
    sm.create_training_job(
        TrainingJobName=job,
        RoleArn=a.role_arn,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File",
                                "ContainerEntrypoint": ["bash", "/opt/ml/input/data/code/entry.sh"]},
        Environment={"EPOCHS": str(a.epochs)},
        InputDataConfig=[
            {"ChannelName": "code", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": f"s3://{a.bucket}/pam/training/{job}/code/",
                "S3DataDistributionType": "FullyReplicated"}}},
            {"ChannelName": "tiles", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": f"s3://{a.bucket}/{a.tiles_prefix}",
                "S3DataDistributionType": "FullyReplicated"}}},
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{a.bucket}/pam/training/{job}/output/"},
        ResourceConfig={"InstanceType": a.instance, "InstanceCount": 1, "VolumeSizeInGB": 50},
        StoppingCondition={"MaxRuntimeInSeconds": 6 * 3600},
    )
    print(f"started {job}; follow it in the SageMaker console or with "
          f"aws sagemaker describe-training-job --training-job-name {job}")


if __name__ == "__main__":
    main()

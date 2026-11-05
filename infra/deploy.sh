#!/usr/bin/env bash
# Build, push and deploy the PAM prototype to AWS (ECS Fargate + ALB + S3).
#
# Requirements: AWS CLI v2 (credentials for the target account), Docker, bash.
# Usage:
#   export PAM_API_KEYS="$(openssl rand -hex 24)"        # keep this; clients use it
#   ./infra/deploy.sh [stack-name] [region] [image-tag]
#
# Optional env: CERT_ARN (ACM certificate for HTTPS), ALLOWED_CIDR, DESIRED_COUNT, ALARM_EMAIL
set -euo pipefail

STACK="${1:-pam-prototype}"
REGION="${2:-${AWS_REGION:-us-east-1}}"
TAG="${3:-$(date +%Y%m%d-%H%M%S)}"
REPO="${STACK}-api"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${PAM_API_KEYS:?Set PAM_API_KEYS to one or more comma-separated keys (16+ chars)}"
[ -f "$ROOT/models/pam_unet.onnx" ] || [ -f "$ROOT/models/pam_unet_int8w.onnx" ] || { echo "no model in models/ (run scripts/export_onnx.py)"; exit 1; }

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"
echo "Account ${ACCOUNT} · region ${REGION} · stack ${STACK} · image ${IMAGE}"

# 1. ECR repository (scan on push, keep last 10 images)
if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true --encryption-configuration encryptionType=AES256 >/dev/null
  aws ecr put-lifecycle-policy --repository-name "$REPO" --region "$REGION" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' >/dev/null
fi

# 2. Build and push (linux/amd64 to match the Fargate task)
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 -t "$IMAGE" "$ROOT"
docker push "$IMAGE"

# 3. Network: default VPC and its public subnets unless VPC_ID / SUBNET_IDS are given
VPC_ID="${VPC_ID:-$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)}"
SUBNET_IDS="${SUBNET_IDS:-$(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text | tr '\t' ',')}"
[ "$VPC_ID" != "None" ] || { echo "No default VPC: set VPC_ID and SUBNET_IDS"; exit 1; }

# 4. Deploy the stack
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
  --template-file "$ROOT/infra/cloudformation.yaml" --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides \
    ImageUri="$IMAGE" VpcId="$VPC_ID" SubnetIds="$SUBNET_IDS" ApiKeys="$PAM_API_KEYS" \
    CertificateArn="${CERT_ARN:-}" AllowedCidr="${ALLOWED_CIDR:-0.0.0.0/0}" \
    DesiredCount="${DESIRED_COUNT:-2}" AlarmEmail="${ALARM_EMAIL:-}"

URL="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)"
echo "Waiting for the service to report ready at ${URL}/v1/ready ..."
for i in $(seq 1 40); do
  if curl -fsS "${URL}/v1/ready" >/dev/null 2>&1; then echo "Ready: ${URL}  (API docs: ${URL}/docs)"; exit 0; fi
  sleep 15
done
echo "Stack deployed but not ready yet; check CloudWatch logs group /ecs/${STACK}"; exit 1

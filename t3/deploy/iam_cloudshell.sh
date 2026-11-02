#!/usr/bin/env bash
# IAM for Team13's Track 3 tools. Run it in AWS CloudShell (us-east-2) as the team user:
#
#   EVIDENCE_BUCKET=<team us-east-2 bucket> bash iam_cloudshell.sh
#
# It is idempotent (safe to run again) and prints every change. It contains no secrets; the
# account id is read at run time. It creates or updates:
#   1. role /bb26/team13-ran-tools   (Lambda; event boundary) + inline policy team13-ran-tools
#   2. role /bb26/team13-trust-gate  (Lambda; event boundary) + inline policy team13-trust-gate
#   3. inline policy team13-ran-tools-invoke on the kit's role team13-gateway
#      (separately named, so the kit's deploy, which rewrites only its own policy, keeps it)
# Run it after the kit's deploy (the Gateway role and the policy engine must exist).
#
# Optional settings (environment):
#   TEAM=team13  REGION=us-east-2
#   STANDIN_MODEL=openai.gpt-oss-20b-1:0     Bedrock stand-in for the NLM
#   NLM_ENDPOINT=team13-otel-nlm             our SageMaker endpoint name
#   ENGINE_ID=<policy engine id>             else looked up by name <team>_policies
set -euo pipefail

TEAM="${TEAM:-team13}"
REGION="${REGION:-us-east-2}"
STANDIN_MODEL="${STANDIN_MODEL:-openai.gpt-oss-20b-1:0}"
NLM_ENDPOINT="${NLM_ENDPOINT:-${TEAM}-otel-nlm}"
: "${EVIDENCE_BUCKET:?set EVIDENCE_BUCKET to the team us-east-2 bucket name}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BOUNDARY="arn:aws:iam::${ACCOUNT}:policy/bb26-boundary/bb26-Boundary"
TABLE_ARN="arn:aws:dynamodb:${REGION}:${ACCOUNT}:table/${TEAM}-approvals"
TOOLS_FN="${TEAM}-ran-tools"
TRUST_FN="${TEAM}-trust-gate"

if [ -z "${ENGINE_ID:-}" ]; then
  ENGINE_ID="$(aws bedrock-agentcore-control list-policy-engines --region "$REGION" \
      --query "policyEngines[?name=='${TEAM}_policies'].policyEngineId | [0]" --output text 2>/dev/null || true)"
fi
if [ -z "$ENGINE_ID" ] || [ "$ENGINE_ID" = "None" ]; then
  echo "Could not find the policy engine ${TEAM}_policies: run the kit's deploy first, or set ENGINE_ID." >&2
  exit 1
fi
ENGINE_ARN="arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT}:policy-engine/${ENGINE_ID}"

echo "account ${ACCOUNT}  region ${REGION}  team ${TEAM}  engine ${ENGINE_ID}  bucket ${EVIDENCE_BUCKET}"
aws iam get-policy --policy-arn "$BOUNDARY" >/dev/null \
  || { echo "The event boundary ${BOUNDARY} is missing." >&2; exit 1; }

TRUST_DOC="$(cat <<EOF
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
  "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": "${ACCOUNT}"}}}]}
EOF
)"

ensure_role() {  # name
  local name="$1"
  if aws iam get-role --role-name "$name" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$name" --policy-document "$TRUST_DOC"
    aws iam put-role-permissions-boundary --role-name "$name" --permissions-boundary "$BOUNDARY"
    echo "role ${name}: exists; trust policy and boundary refreshed"
  else
    aws iam create-role --role-name "$name" --path /bb26/ --permissions-boundary "$BOUNDARY" \
      --assume-role-policy-document "$TRUST_DOC" \
      --description "Team13 Track 3 ${name} Lambda" >/dev/null
    echo "role ${name}: created under /bb26/ with the event boundary"
  fi
}

put_policy() {  # role, policy name, document
  aws iam put-role-policy --role-name "$1" --policy-name "$2" --policy-document "$3"
  echo "inline policy ${2} on role ${1}: written"
}

logs_statement() {  # function name
  cat <<EOF
{"Sid": "OwnLogs", "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
 "Resource": ["arn:aws:logs:${REGION}:${ACCOUNT}:log-group:/aws/lambda/$1",
              "arn:aws:logs:${REGION}:${ACCOUNT}:log-group:/aws/lambda/$1:*"]}
EOF
}

# 1. the tools: approvals table (approval rows chg-*, case rows t3-*), evidence, trust gate, NLM
ensure_role "$TOOLS_FN"
put_policy "$TOOLS_FN" "$TOOLS_FN" "$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Sid": "ApprovalsAndCases", "Effect": "Allow",
  "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"], "Resource": "${TABLE_ARN}",
  "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["chg-*", "t3-*"]}}},
 {"Sid": "Evidence", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"],
  "Resource": "arn:aws:s3:::${EVIDENCE_BUCKET}/evidence/*"},
 {"Sid": "TrustGateInvoke", "Effect": "Allow", "Action": "lambda:InvokeFunction",
  "Resource": "arn:aws:lambda:${REGION}:${ACCOUNT}:function:${TRUST_FN}"},
 {"Sid": "StandInNlm", "Effect": "Allow", "Action": "bedrock:InvokeModel",
  "Resource": ["arn:aws:bedrock:*::foundation-model/${STANDIN_MODEL}",
               "arn:aws:bedrock:${REGION}:${ACCOUNT}:inference-profile/*${STANDIN_MODEL}"]},
 {"Sid": "TunedNlm", "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
  "Resource": "arn:aws:sagemaker:${REGION}:${ACCOUNT}:endpoint/${NLM_ENDPOINT}"},
 $(logs_statement "$TOOLS_FN")
]}
EOF
)"

# 2. the trust gate: pause rules on OUR policy engine only, its trust rows, control records, logs
ensure_role "$TRUST_FN"
put_policy "$TRUST_FN" "$TRUST_FN" "$(cat <<EOF
{"Version": "2012-10-17", "Statement": [
 {"Sid": "PausePoliciesOnOurEngineOnly", "Effect": "Allow",
  "Action": ["bedrock-agentcore:CreatePolicy", "bedrock-agentcore:DeletePolicy",
             "bedrock-agentcore:GetPolicy", "bedrock-agentcore:ListPolicies"],
  "Resource": ["${ENGINE_ARN}", "${ENGINE_ARN}/*"]},
 {"Sid": "TrustRows", "Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
  "Resource": "${TABLE_ARN}",
  "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["t3-trust#*"]}}},
 {"Sid": "ControlRecords", "Effect": "Allow", "Action": "s3:PutObject",
  "Resource": "arn:aws:s3:::${EVIDENCE_BUCKET}/evidence/records/*"},
 $(logs_statement "$TRUST_FN")
]}
EOF
)"

# 3. let the kit's Gateway role invoke our tools Lambda (a policy the kit never rewrites)
if aws iam get-role --role-name "${TEAM}-gateway" >/dev/null 2>&1; then
  put_policy "${TEAM}-gateway" "${TEAM}-ran-tools-invoke" "$(cat <<EOF
{"Version": "2012-10-17", "Statement": [{"Sid": "InvokeTeam13RanTools", "Effect": "Allow",
  "Action": "lambda:InvokeFunction",
  "Resource": ["arn:aws:lambda:${REGION}:${ACCOUNT}:function:${TOOLS_FN}",
               "arn:aws:lambda:${REGION}:${ACCOUNT}:function:${TOOLS_FN}:*"]}]}
EOF
)"
else
  echo "role ${TEAM}-gateway not found: run the kit's deploy first, then run this script again." >&2
  exit 1
fi

echo "done. Roles:"
aws iam get-role --role-name "$TOOLS_FN" --query Role.Arn --output text
aws iam get-role --role-name "$TRUST_FN" --query Role.Arn --output text
echo "Next (laptop): python -m t3.deploy.register --kit-dir <kit folder> --image-uri <ecr image uri>   (dry run)"

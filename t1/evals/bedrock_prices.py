"""Bedrock on-demand token prices from the AWS Price List, and the decoding sent per Bedrock model.

    python -m t1.evals.bedrock_prices --model-id openai.gpt-oss-20b-1:0 [--from-file products.json]

prints the harness flags `--price-in-per-1m X --price-out-per-1m Y --price-source "..."`. It reads
the AWS Price List (pricing:GetProducts, read-only, us-east-1 endpoint) or a saved
`aws pricing get-products --service-code AmazonBedrock` output, and FAILS CLOSED: exactly one
standard on-demand input price and one output price for the model in the region, in USD per 1K
tokens, or it exits non-zero. Batch, flex, priority, cache and "mantle" usage types never match.

Prices read on 2026-11-02 (effective 2026-10-01), us-east-2, standard on-demand, USD per 1M tokens:
  openai.gpt-oss-20b-1:0   input 0.07, output 0.30   (USE2-gpt-oss-20b-{input,output}-tokens)
  qwen.qwen3-32b-v1:0      input 0.15, output 0.60   (USE2-Qwen3-32B-{input,output}-tokens)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

REGION_PREFIX = {"us-east-2": "USE2", "us-east-1": "USE1", "us-west-2": "USW2"}
# Bedrock model id -> the model's name inside Price List usage types
USAGE_NAME = {"openai.gpt-oss-20b-1:0": "gpt-oss-20b", "qwen.qwen3-32b-v1:0": "Qwen3-32B"}

# What the harness sends per model to keep reasoning as short as the API allows (recorded in decoding).
DECODING = {
    "openai.gpt-oss-20b-1:0": {
        "additional_fields": {"reasoning_effort": "low"},
        "thinking": "gpt-oss always reasons; reasoning_effort=low is the lowest setting (OpenAI chat-completion "
                    "field passed through Converse additionalModelRequestFields). Reasoning comes back as "
                    "reasoningContent, is not part of the answer text, and its tokens are in outputTokens (billed).",
    },
    "qwen.qwen3-32b-v1:0": {
        "additional_fields": {},
        "thinking": "no documented request field turns Qwen3 thinking off on Bedrock (its model card's Converse "
                    "example has none; the /no_think prompt switch would change the shared prompt, so it is not "
                    "used): measured as served. Reasoning comes back as reasoningContent, is not part of the "
                    "answer text, and its tokens are in outputTokens (billed).",
    },
}


class PriceNotFound(RuntimeError):
    pass


def _products(items: list) -> list[dict]:
    return [json.loads(p) if isinstance(p, str) else p for p in items]


def lookup(price_list: list, model_id: str, region: str = "us-east-2") -> dict:
    """{in_per_1m, out_per_1m, usagetypes, effective} for one model, or PriceNotFound."""
    if model_id not in USAGE_NAME:
        raise PriceNotFound(f"no Price List name on file for {model_id}; add it to USAGE_NAME")
    prefix = f"{REGION_PREFIX[region]}-{USAGE_NAME[model_id]}"
    out = {}
    for direction in ("input", "output"):
        want = f"{prefix}-{direction}-tokens"
        hits = []
        for p in _products(price_list):
            a = p["product"]["attributes"]
            if a.get("usagetype") != want or a.get("regionCode", region) != region:
                continue
            for term in p.get("terms", {}).get("OnDemand", {}).values():
                for dim in term["priceDimensions"].values():
                    if dim.get("unit") == "1K tokens" and "USD" in dim.get("pricePerUnit", {}):
                        hits.append((float(dim["pricePerUnit"]["USD"]), term.get("effectiveDate"), dim.get("description")))
        if len(hits) != 1:
            raise PriceNotFound(f"{model_id}: expected exactly one on-demand '{want}' price per 1K tokens, found {len(hits)}")
        out[direction] = hits[0]
    return {"in_per_1m": round(out["input"][0] * 1000, 10), "out_per_1m": round(out["output"][0] * 1000, 10),
            "usagetypes": [f"{prefix}-input-tokens", f"{prefix}-output-tokens"],
            "effective": out["input"][1], "descriptions": [out["input"][2], out["output"][2]]}


def fetch(region: str, profile: str | None = None) -> list:
    """All AmazonBedrock products for the region from the Price List API (read-only)."""
    import boto3

    c = boto3.Session(profile_name=profile).client("pricing", region_name="us-east-1")
    items, kw = [], {"ServiceCode": "AmazonBedrock",
                     "Filters": [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}]}
    for page in c.get_paginator("get_products").paginate(**kw):
        items += page["PriceList"]
    return items


def source_text(model_id: str, info: dict, region: str, read_on: str) -> str:
    return (f"AWS Price List API, AmazonBedrock, {region}, standard on-demand, usagetypes "
            f"{' / '.join(info['usagetypes'])}, effective {(info['effective'] or '?')[:10]}, read {read_on}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--from-file", help="saved `aws pricing get-products` JSON ({PriceList: [...]})")
    ap.add_argument("--read-on", default=date.today().isoformat(), help="date the price list was read")
    ap.add_argument("--profile")
    a = ap.parse_args(argv)
    items = json.load(open(a.from_file))["PriceList"] if a.from_file else fetch(a.region, a.profile)
    try:
        info = lookup(items, a.model_id, a.region)
    except PriceNotFound as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    src = source_text(a.model_id, info, a.region, a.read_on)
    print(f"--price-in-per-1m {info['in_per_1m']:g} --price-out-per-1m {info['out_per_1m']:g} --price-source '{src}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

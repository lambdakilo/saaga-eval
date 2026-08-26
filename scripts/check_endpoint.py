#!/usr/bin/env python3
"""Validate a model endpoint before spending a benchmark run on it.

Reads the API key from a file or environment variable and never prints it --
only its length and a masked prefix, so a wrong-file mistake is still visible.

Catches the failures that otherwise surface an hour into a run:

* a model listed in the catalogue but not actually deployed (HTTP 404)
* an endpoint so slow that an agentic loop will time out
* a reasoning model whose `reasoning_content` eats the whole completion budget,
  leaving empty `content` -- which looks like a broken harness, not a
  misconfigured token limit

Usage::

    python scripts/check_endpoint.py --key-file secret --model moonshotai/kimi-k3
    python scripts/check_endpoint.py --list | grep -i qwen
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def load_key(key_file: Path | None, env_var: str) -> str:
    """Read a key from a file or the environment, tolerating `VAR=value` form."""
    if key_file:
        raw = key_file.read_text(encoding="utf-8").strip()
    else:
        raw = os.getenv(env_var, "").strip()
    if not raw:
        raise SystemExit(
            f"No key found. Pass --key-file, or export {env_var}."
        )
    if "=" in raw.split("\n")[0] and not raw.startswith("nvapi-"):
        raw = raw.split("=", 1)[1]
    return raw.strip().strip("\"'")


def request(url: str, key: str, payload: dict | None = None) -> tuple[int, dict, float]:
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST" if payload else "GET",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.status, json.loads(response.read()), time.time() - start
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = {"raw": body[:300]}
        return exc.code, parsed, time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key-file", type=Path, help="File holding the API key (never printed)")
    parser.add_argument("--env-var", default="NVIDIA_NIM_API_KEY")
    parser.add_argument("--base-url", default=NIM_BASE_URL)
    parser.add_argument("--model", help="Model id to probe")
    parser.add_argument("--list", action="store_true", help="List available model ids and exit")
    parser.add_argument("--max-tokens", type=int, default=256, help="Budget for the probe completion")
    args = parser.parse_args()

    key = load_key(args.key_file, args.env_var)
    print(f"key: {len(key)} chars, {key[:6]}{'*' * 8}")

    status, body, _ = request(f"{args.base_url}/models", key)
    if status != 200:
        raise SystemExit(f"Auth or endpoint failure (HTTP {status}): {str(body)[:200]}")
    ids = sorted(m["id"] for m in body.get("data", []))
    print(f"endpoint OK: {len(ids)} models listed")

    if args.list:
        for model_id in ids:
            print(" ", model_id)
        return 0

    if not args.model:
        raise SystemExit("Pass --model to probe one, or --list to see what is available.")

    if args.model not in ids:
        print(f"\nWARNING: {args.model!r} is not in the catalogue listing.")

    status, body, elapsed = request(
        f"{args.base_url}/chat/completions",
        key,
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
    )

    if status != 200:
        detail = body.get("detail") or body
        raise SystemExit(
            f"\nFAIL  {args.model}  HTTP {status}\n  {str(detail)[:300]}\n"
            "  A catalogue listing does not guarantee the model is deployed."
        )

    message = body["choices"][0].get("message", {})
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    usage = body.get("usage", {})

    print(f"\nOK    {args.model}")
    print(f"  latency        : {elapsed:.1f}s")
    print(f"  content        : {content[:80]!r}")
    print(f"  completion toks: {usage.get('completion_tokens')}")

    if reasoning:
        print(f"  reasoning model: yes ({len(reasoning)} chars of reasoning_content)")
        if not content:
            print(
                "\n  WARNING: empty content with non-empty reasoning.\n"
                f"  The {args.max_tokens}-token budget was consumed before an answer.\n"
                "  Raise the agent's completion budget or the run will look broken."
            )
    if elapsed > 30:
        print(
            f"\n  WARNING: {elapsed:.0f}s for a trivial prompt. An agentic loop makes\n"
            "  dozens of calls; this endpoint will be painfully slow."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Classify ABPM free-text note fields into a binary target using Ollama.

Input:
- Summary CSV produced by extract_data.py (must include diagnostics and/or contents columns)

Output:
- A new CSV containing all original columns plus:
    - health_label (healthy or not)

This script is a text-based helper and does not provide medical advice.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

VALID_LABELS = {"healthy", "not"}

SYSTEM_INSTRUCTION = (
    "You classify ABPM report note text into one binary target. "
    "Return exactly one label: healthy or not. "
    "Rules: healthy only if the note clearly indicates a normal BP profile and no abnormalities. "
    "If any abnormality, hypertension concern, uncertainty, or insufficient information, return not."
)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(raw: str) -> str | None:
    text = normalize_ws(raw).lower()
    if text == "healthy":
        return "healthy"
    if text == "not":
        return "not"
    return None


def call_ollama(note_text: str, model: str, ollama_url: str, timeout_seconds: int, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": build_prompt(note_text),
        "stream": False,
        "options": {"temperature": temperature},
    }

    body = json.dumps(payload).encode("utf-8")
    request = Request(
        ollama_url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))

    raw_response = str(data.get("response", "")).strip()

    label = normalize_label(raw_response)
    if label is None:
        raise ValueError(
            f"Model must return only 'healthy' or 'not'. Got: {raw_response!r}"
        )
    return label


def build_prompt(note_text: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        "ABPM note text:\n"
        f"{note_text}\n\n"
        "Output one label only: healthy or not"
    )


def classify_row(
    row: dict[str, str],
    text_field: str,
    model: str,
    ollama_url: str,
    timeout_seconds: int,
    temperature: float,
) -> str:
    note_text = normalize_ws(row.get(text_field, ""))
    return call_ollama(
        note_text=note_text,
        model=model,
        ollama_url=ollama_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add binary healthy/not target to ABPM summary CSV using Ollama."
    )
    parser.add_argument(
        "--input-csv",
        default="data/bp_summary.csv",
        help="Input summary CSV path (default: data/bp_summary.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default="data/bp_summary_classified.csv",
        help="Output CSV path (default: data/bp_summary_classified.csv)",
    )
    parser.add_argument(
        "--text-field",
        default="diagnostics",
        help="Single source text column for LLM classification (default: diagnostics)",
    )
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (default: 0.0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        input_fields = reader.fieldnames or []

    if args.text_field not in input_fields:
        raise SystemExit(f"Input CSV is missing required text field: {args.text_field}")

    out_fields = input_fields.copy()
    if "health_label" not in out_fields:
        out_fields.append("health_label")

    total_rows = len(rows)
    print(f"Starting classification for {total_rows} rows...", flush=True)
    start_time = tpime.time()

    for index, row in enumerate(rows, start=1):
        source_name = row.get("source_file", "unknown")
        print(f"[{index}/{total_rows}] Classifying: {source_name}", flush=True)
        try:
            label = classify_row(
                row=row,
                text_field=args.text_field,
                model=args.model,
                ollama_url=args.ollama_url,
                timeout_seconds=args.timeout,
                temperature=args.temperature,
            )
        except (HTTPError, OSError, ValueError) as exc:
            raise SystemExit(f"Failed classifying row '{source_name}': {exc}")

        if label not in VALID_LABELS:
            raise SystemExit(f"Unexpected label from Ollama: {label!r}. Expected one of: {sorted(VALID_LABELS)}")
        row["health_label"] = label

        if index % 10 == 0 or index == total_rows:
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0.0
            eta_seconds = int((total_rows - index) / rate) if rate > 0 else 0
            print(
                f"Progress: {index}/{total_rows} done | Elapsed: {elapsed:.1f}s | ETA: {eta_seconds}s",
                flush=True,
            )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.time() - start_time
    print(f"Input rows: {len(rows)}")
    print(f"Output CSV: {output_path.resolve()}")
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

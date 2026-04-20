"""Extract structured ABPM data from 4-page blood pressure report PDFs.

The script reads either one PDF file or a directory of PDFs, then extracts:
- Page 1 summary fields (age, sex, DOB, averages, loads, max/min, circadian, BP CV, AASI)
- Page 4 measurement table rows

Outputs:
- Summary CSV: one row per PDF
- Readings CSV: one row per measurement row from page 4
- JSON: full structured payload for each PDF
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


DATE_TIME_TOKEN_RE = re.compile(r"^(\d{4}/\d{1,2}/\d{1,2})\s*(\d{1,2}:\d{2})$")
TABLE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d{4}/\d{1,2}/\d{1,2})\s*(\d{1,2}:\d{2})\s+"
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
    r"(?:\s+(.*))?\s*$"
)

SUMMARY_FIELDS = [
    "source_file",
    "test_begin",
    "age",
    "sex",
    "date_of_birth",
    "all_avg_sys",
    "all_avg_dia",
    "day_avg_sys",
    "day_avg_dia",
    "day_bp_threshold_sys",
    "day_bp_threshold_dia",
    "night_avg_sys",
    "night_avg_dia",
    "night_bp_threshold_sys",
    "night_bp_threshold_dia",
    "day_bp_load_value_rule",
    "night_bp_load_value_rule",
    "day_sys_load_threshold",
    "day_sys_load_pct",
    "night_sys_load_threshold",
    "night_sys_load_pct",
    "day_dia_load_threshold",
    "day_dia_load_pct",
    "night_dia_load_threshold",
    "night_dia_load_pct",
    "max_sys",
    "max_sys_datetime",
    "min_sys",
    "min_sys_datetime",
    "max_dia",
    "max_dia_datetime",
    "min_dia",
    "min_dia_datetime",
    "circadian_sys_night_des_pct",
    "circadian_dia_night_des_pct",
    "circadian_normal_range",
    "bp_cv_all_sys_pct",
    "bp_cv_all_dia_pct",
    "bp_cv_day_sys_pct",
    "bp_cv_day_dia_pct",
    "bp_cv_night_sys_pct",
    "bp_cv_night_dia_pct",
    "contents",
    "diagnostics",
    "aasi_sys",
    "readings_count",
]

READING_FIELDS = [
    "source_file",
    "num",
    "date",
    "time",
    "datetime",
    "sys",
    "map",
    "dia",
    "pp",
    "pr",
    "state",
    "comment",
]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_datetime_token(value: str) -> str:
    value = value.strip()
    match = DATE_TIME_TOKEN_RE.match(value)
    if not match:
        return value
    return f"{match.group(1)} {match.group(2)}"


def to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def clean_text_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = normalize_ws(value)
    return cleaned or None


def extract_contents_diagnostics(page1_text: str) -> tuple[str | None, str | None]:
    # First try a normalized-text regex: this handles cases where PDF extraction flattens lines.
    flat_text = normalize_ws(page1_text)

    contents_match = re.search(
        r"(?:Contents?|Comments?)\s*[:\-]?\s*(.+?)(?=\s*Diagnostics?\s*[:\-]|$)",
        flat_text,
        flags=re.IGNORECASE,
    )
    diagnostics_match = re.search(
        r"Diagnostics?\s*[:\-]?\s*(.+)$",
        flat_text,
        flags=re.IGNORECASE,
    )

    contents = clean_text_value(contents_match.group(1)) if contents_match else None
    diagnostics = clean_text_value(diagnostics_match.group(1)) if diagnostics_match else None

    # Fallback: parse line-by-line when labels are on separate lines or in a boxed section.
    if contents is not None or diagnostics is not None:
        return contents, diagnostics

    contents_parts: list[str] = []
    diagnostics_parts: list[str] = []
    mode: str | None = None

    for raw_line in page1_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_contents = re.match(r"^(?:Contents?|Comments?)\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if m_contents:
            mode = "contents"
            remainder = m_contents.group(1).strip()
            if remainder:
                contents_parts.append(remainder)
            continue

        m_diagnostics = re.match(r"^Diagnostics?\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if m_diagnostics:
            mode = "diagnostics"
            remainder = m_diagnostics.group(1).strip()
            if remainder:
                diagnostics_parts.append(remainder)
            continue

        if mode == "contents":
            contents_parts.append(line)
        elif mode == "diagnostics":
            diagnostics_parts.append(line)

    line_based_contents = clean_text_value(" ".join(contents_parts))
    line_based_diagnostics = clean_text_value(" ".join(diagnostics_parts))
    if line_based_contents is not None or line_based_diagnostics is not None:
        return line_based_contents, line_based_diagnostics

    # Last fallback: infer the bottom boxed section after AASI when labels are absent.
    lines = [ln.strip() for ln in page1_text.splitlines()]
    aasi_index = next((i for i, ln in enumerate(lines) if re.search(r"\bAASI\b", ln, flags=re.IGNORECASE)), None)
    if aasi_index is None:
        return None, None

    tail_lines = [ln for ln in lines[aasi_index + 1 :] if ln]
    filtered_tail: list[str] = []
    for ln in tail_lines:
        if re.search(r"doctor\s+assistant\s+date", ln, flags=re.IGNORECASE):
            continue
        if re.search(r"used\s+for\s+clinical\s+reference", ln, flags=re.IGNORECASE):
            continue
        filtered_tail.append(ln)

    if not filtered_tail:
        return None, None

    diagnostics = clean_text_value(filtered_tail[0])
    contents = clean_text_value(" ".join(filtered_tail[1:])) if len(filtered_tail) > 1 else None
    return contents, diagnostics


def parse_summary(page1_text: str, source_file: str) -> dict[str, Any]:
    text = normalize_ws(page1_text)

    summary: dict[str, Any] = {field: None for field in SUMMARY_FIELDS}
    summary["source_file"] = source_file

    m = re.search(
        r"Test Begin:\s*(\d{4}/\d{1,2}/\d{1,2}\s*\d{1,2}:\d{2}:\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["test_begin"] = normalize_datetime_token(m.group(1))

    m = re.search(r"Age:\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        summary["age"] = to_int(m.group(1))

    m = re.search(r"Male/Female:\s*(Male|Female)", text, flags=re.IGNORECASE)
    if m:
        summary["sex"] = m.group(1).title()

    m = re.search(r"Date of Birth:\s*(\d{4}/\d{1,2}/\d{1,2})", text, flags=re.IGNORECASE)
    if m:
        summary["date_of_birth"] = m.group(1)

    m = re.search(r"All BP Averages:\s*([\d.]+)\s*/\s*([\d.]+)\s*mmHg", text, flags=re.IGNORECASE)
    if m:
        summary["all_avg_sys"] = to_float(m.group(1))
        summary["all_avg_dia"] = to_float(m.group(2))

    m = re.search(
        r"Day BP Averages:\s*([\d.]+)\s*/\s*([\d.]+)\s*mmHg\s*BP threshold:\s*(\d+)\s*/\s*(\d+)\s*mmHg",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["day_avg_sys"] = to_float(m.group(1))
        summary["day_avg_dia"] = to_float(m.group(2))
        summary["day_bp_threshold_sys"] = to_int(m.group(3))
        summary["day_bp_threshold_dia"] = to_int(m.group(4))

    m = re.search(
        r"Night BP Averages:\s*([\d.]+)\s*/\s*([\d.]+)\s*mmHg\s*BP threshold:\s*(\d+)\s*/\s*(\d+)\s*mmHg",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["night_avg_sys"] = to_float(m.group(1))
        summary["night_avg_dia"] = to_float(m.group(2))
        summary["night_bp_threshold_sys"] = to_int(m.group(3))
        summary["night_bp_threshold_dia"] = to_int(m.group(4))

    m = re.search(
        r"Day BP Load Value:\s*([^\s]+)\s*Night BP Load Value:\s*([^\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["day_bp_load_value_rule"] = m.group(1)
        summary["night_bp_load_value_rule"] = m.group(2)

    m = re.search(
        r"SYS\(>(\d+)mmHg\)\s*([\d.]+)%\s*SYS\(>(\d+)mmHg\)\s*([\d.]+)%",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["day_sys_load_threshold"] = to_int(m.group(1))
        summary["day_sys_load_pct"] = to_float(m.group(2))
        summary["night_sys_load_threshold"] = to_int(m.group(3))
        summary["night_sys_load_pct"] = to_float(m.group(4))

    m = re.search(
        r"DIA\(>(\d+)mmHg\)\s*([\d.]+)%\s*DIA\(>(\d+)mmHg\)\s*([\d.]+)%",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["day_dia_load_threshold"] = to_int(m.group(1))
        summary["day_dia_load_pct"] = to_float(m.group(2))
        summary["night_dia_load_threshold"] = to_int(m.group(3))
        summary["night_dia_load_pct"] = to_float(m.group(4))

    dt_pattern = r"(\d{4}/\d{1,2}/\d{1,2}\s*\d{1,2}:\d{2})"
    m = re.search(
        rf"Maximum SYS\s*(\d+)\s*mmHg\s*on\s*{dt_pattern}\s*Minimum SYS\s*(\d+)\s*mmHg\s*on\s*{dt_pattern}",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["max_sys"] = to_int(m.group(1))
        summary["max_sys_datetime"] = normalize_datetime_token(m.group(2))
        summary["min_sys"] = to_int(m.group(3))
        summary["min_sys_datetime"] = normalize_datetime_token(m.group(4))

    m = re.search(
        rf"Maximum DIA\s*(\d+)\s*mmHg\s*on\s*{dt_pattern}\s*Minimum DIA\s*(\d+)\s*mmHg\s*on\s*{dt_pattern}",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["max_dia"] = to_int(m.group(1))
        summary["max_dia_datetime"] = normalize_datetime_token(m.group(2))
        summary["min_dia"] = to_int(m.group(3))
        summary["min_dia_datetime"] = normalize_datetime_token(m.group(4))

    m = re.search(
        r"Circadian rhythm of BP:\s*SYS Night Des\.?\s*([\d.]+)%\s*DIA Night Des\.?\s*([\d.]+)%\s*Normal:\s*([\d]+%-[\d]+%)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["circadian_sys_night_des_pct"] = to_float(m.group(1))
        summary["circadian_dia_night_des_pct"] = to_float(m.group(2))
        summary["circadian_normal_range"] = m.group(3)

    m = re.search(
        r"BP CV:\s*All:SYS\s*([\d.]+)%\s*DIA\s*([\d.]+)%\s*Day:SYS\s*([\d.]+)%\s*DIA\s*([\d.]+)%\s*Night:SYS\s*([\d.]+)%\s*DIA\s*([\d.]+)%",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        summary["bp_cv_all_sys_pct"] = to_float(m.group(1))
        summary["bp_cv_all_dia_pct"] = to_float(m.group(2))
        summary["bp_cv_day_sys_pct"] = to_float(m.group(3))
        summary["bp_cv_day_dia_pct"] = to_float(m.group(4))
        summary["bp_cv_night_sys_pct"] = to_float(m.group(5))
        summary["bp_cv_night_dia_pct"] = to_float(m.group(6))

    m = re.search(r"AASI:\s*SYS\s*([\d.]+)", text, flags=re.IGNORECASE)
    if m:
        summary["aasi_sys"] = to_float(m.group(1))

    contents, diagnostics = extract_contents_diagnostics(page1_text)
    summary["contents"] = contents
    summary["diagnostics"] = diagnostics

    return summary


def parse_readings(page4_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line in page4_text.splitlines():
        match = TABLE_ROW_RE.match(line)
        if not match:
            continue

        date_token = match.group(2)
        time_token = match.group(3)

        row = {
            "num": to_int(match.group(1)),
            "date": date_token,
            "time": time_token,
            "datetime": normalize_datetime_token(f"{date_token} {time_token}"),
            "sys": to_int(match.group(4)),
            "map": to_int(match.group(5)),
            "dia": to_int(match.group(6)),
            "pp": to_int(match.group(7)),
            "pr": to_int(match.group(8)),
            "state": to_int(match.group(9)),
            "comment": (match.group(10) or "").strip(),
        }
        rows.append(row)

    return rows


def parse_report(pdf_path: Path) -> dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) < 4:
        raise ValueError(f"expected at least 4 pages, got {len(reader.pages)}")

    page1_text = reader.pages[0].extract_text() or ""
    page4_text = reader.pages[3].extract_text() or ""

    summary = parse_summary(page1_text, source_file=pdf_path.name)
    readings = parse_readings(page4_text)
    summary["readings_count"] = len(readings)

    return {
        "source_file": pdf_path.name,
        "summary": summary,
        "readings": readings,
    }


def collect_pdf_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"input file is not a PDF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"input path does not exist: {input_path}")

    return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract page-1 summary metrics and page-4 table readings from ABPM report PDFs."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=".",
        help="PDF file path or directory containing PDFs (default: current directory)",
    )
    parser.add_argument(
        "--summary-csv",
        default="bp_summary.csv",
        help="Output CSV for summary rows (default: bp_summary.csv)",
    )
    parser.add_argument(
        "--readings-csv",
        default="bp_readings.csv",
        help="Output CSV for page-4 table rows (default: bp_readings.csv)",
    )
    parser.add_argument(
        "--json-out",
        default="bp_extracted.json",
        help="Output JSON with full parsed structure (default: bp_extracted.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    pdf_files = collect_pdf_files(input_path)
    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {input_path}")

    reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reading_rows: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        try:
            report = parse_report(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to parse {pdf_path.name}: {exc}")
            continue

        reports.append(report)
        summary_rows.append(report["summary"])

        for row in report["readings"]:
            reading_rows.append({"source_file": pdf_path.name, **row})

    if not reports:
        raise SystemExit("No PDF files were parsed successfully.")

    summary_csv_path = Path(args.summary_csv)
    readings_csv_path = Path(args.readings_csv)
    json_path = Path(args.json_out)

    write_csv(summary_csv_path, summary_rows, SUMMARY_FIELDS)
    write_csv(readings_csv_path, reading_rows, READING_FIELDS)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(reports)} PDF(s)")
    print(f"Summary CSV:  {summary_csv_path.resolve()}")
    print(f"Readings CSV: {readings_csv_path.resolve()}")
    print(f"JSON output:  {json_path.resolve()}")


if __name__ == "__main__":
    main()

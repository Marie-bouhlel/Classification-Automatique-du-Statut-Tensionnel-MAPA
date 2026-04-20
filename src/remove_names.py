import fitz  # PyMuPDF
import re
from pathlib import Path

# Input/output folder for batch processing.
input_dir = Path(".")
output_dir = Path("cleaned_pdfs")
output_dir.mkdir(exist_ok=True)

# Regex patterns to redact by full line / text span.
regex_patterns = [
    re.compile(r"Patient\s*ID\s*:.*", re.IGNORECASE),
    re.compile(r"\b\d{8,}\b"),  # IDs like 02111995
]


def add_rect_redaction(page, rect):
    # White fill hides the removed content visually.
    page.add_redact_annot(rect, fill=(1, 1, 1))


def normalize_token(token):
    return re.sub(r"[^a-z0-9]", "", token.lower())


def redact_patient_name_value(page):
    """
    Redact only the value after 'Patient Name:' on the same horizontal line.
    Works even if there are large spaces between words.
    """
    words = page.get_text("words")
    label_rects = page.search_for("Patient Name:")
    if not label_rects:
        label_rects = page.search_for("Patient Name")

    y_tolerance = 3.0
    max_name_width = 260.0

    for label in label_rects:
        right_limit = label.x1 + max_name_width
        for w in words:
            x0, y0, x1, y1, token = w[:5]
            token_norm = normalize_token(token)

            same_row = (y0 <= label.y1 + y_tolerance) and (y1 >= label.y0 - y_tolerance)
            in_value_zone = (x0 > label.x1 + 2) and (x1 <= right_limit)

            # Keep the label area untouched; redact only likely name tokens on the right.
            if same_row and in_value_zone and token_norm and token_norm not in {"patient", "name", "id"}:
                add_rect_redaction(page, fitz.Rect(w[:4]))


def redact_regex_spans(page, compiled_patterns):
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            line_rect = fitz.Rect()
            spans = line.get("spans", [])
            line_text = ""

            for span in spans:
                line_text += span.get("text", "")
                line_rect |= fitz.Rect(span["bbox"])

            if not line_text.strip():
                continue

            # Redact the full line if it matches Patient ID key/value text.
            if compiled_patterns[0].search(line_text):
                add_rect_redaction(page, line_rect)

    # Redact standalone long numeric IDs by word boxes.
    for w in page.get_text("words"):
        word = w[4]
        if compiled_patterns[1].fullmatch(word):
            add_rect_redaction(page, fitz.Rect(w[:4]))


def process_pdf(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)

    for page in doc:
        redact_patient_name_value(page)
        redact_regex_spans(page, regex_patterns)
        page.apply_redactions()

    # deflate/garbage options reduce leftovers in saved file objects.
    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()


pdf_files = [p for p in input_dir.glob("*.pdf") if not p.name.lower().endswith("_cleaned.pdf")]

if not pdf_files:
    print("No PDF files found in the current folder.")
else:
    for index, pdf in enumerate(sorted(pdf_files), start=1):
        out_file = output_dir / f"ID_{index:04d}_cleaned.pdf"
        process_pdf(pdf, out_file)
        print(f"{pdf.name} -> {out_file.name}")

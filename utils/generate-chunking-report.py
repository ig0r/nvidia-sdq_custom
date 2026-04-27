#!/usr/bin/env python3
"""Generate a Word doc visualizing logical chunks alongside their source recursive chunks.

Reads pairs of files from a chunking-output directory:
  - {doc_id}-chunks.json        (recursive intermediate chunks)
  - {doc_id}-logic-chunks.json  (logical chunks with `source_chunk_ids` provenance)

Emits one .docx per pair where each logical chunk is shown in a 2-column table
next to the recursive chunks it was built from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.shared import Cm, Pt, RGBColor


LOGIC_SUFFIX = "-logic-chunks.json"
RECURSIVE_SUFFIX = "-chunks.json"
BODY_FONT_PT = 9.5
LABEL_FONT_PT = 9.0
COL_WIDTH = Cm(9.0)


def discover_doc_ids(input_dir: Path) -> list[str]:
    return sorted(p.name[: -len(LOGIC_SUFFIX)] for p in input_dir.glob(f"*{LOGIC_SUFFIX}"))


def load_pair(input_dir: Path, doc_id: str) -> tuple[dict, dict] | None:
    rec_path = input_dir / f"{doc_id}{RECURSIVE_SUFFIX}"
    log_path = input_dir / f"{doc_id}{LOGIC_SUFFIX}"
    if not rec_path.exists():
        print(f"[skip] {doc_id}: missing {rec_path.name}", file=sys.stderr)
        return None
    if not log_path.exists():
        print(f"[skip] {doc_id}: missing {log_path.name}", file=sys.stderr)
        return None
    return json.loads(rec_path.read_text()), json.loads(log_path.read_text())


def write_label(cell, text: str) -> None:
    p = cell.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(LABEL_FONT_PT)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def write_chunk_text(cell, text: str) -> None:
    """Render multiline chunk text in a cell, preserving paragraph and line breaks."""
    body = cell.add_paragraph()
    for i, line in enumerate(text.split("\n")):
        if i > 0:
            body.add_run().add_break()
        run = body.add_run(line)
        run.font.size = Pt(BODY_FONT_PT)


def build_report(rec_data: dict, log_data: dict, out_path: Path) -> None:
    doc = Document()

    for section in doc.sections:
        section.left_margin = Cm(1.5)
        section.right_margin = Cm(1.5)
        section.top_margin = Cm(1.5)
        section.bottom_margin = Cm(1.5)

    doc_id = log_data.get("doc_id") or rec_data.get("doc_id") or out_path.stem
    rec_by_id = {c["chunk_id"]: c for c in rec_data["texts"]}
    log_chunks = log_data["texts"]

    rec_total_tokens = sum(c.get("tokens", 0) for c in rec_data["texts"])
    log_total_tokens = sum(c.get("tokens", 0) for c in log_chunks)

    doc.add_heading(f"Chunking Report — {doc_id}", level=0)

    summary = doc.add_paragraph()
    parsed_label = summary.add_run("parsed_file: ")
    parsed_label.bold = True
    summary.add_run(str(log_data.get("parsed_file", "")))
    summary.add_run().add_break()
    summary.add_run(f"recursive chunks: {len(rec_data['texts'])} (total tokens: {rec_total_tokens})")
    summary.add_run().add_break()
    summary.add_run(f"logical chunks:   {len(log_chunks)} (total tokens: {log_total_tokens})")

    for idx, lc in enumerate(log_chunks):
        source_ids = lc.get("source_chunk_ids", [])
        heading = (
            f"Logical Chunk {lc['chunk_id']}  ·  tokens={lc.get('tokens', '?')}  "
            f"·  sources={source_ids}"
        )
        doc.add_heading(heading, level=2)

        table = doc.add_table(rows=1, cols=2)
        table.autofit = False
        table.style = "Table Grid"
        for col in table.columns:
            col.width = COL_WIDTH
        for cell in table.rows[0].cells:
            cell.width = COL_WIDTH
            cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP

        left, right = table.rows[0].cells

        write_label(left, f"Logical chunk #{lc['chunk_id']} (tokens={lc.get('tokens', '?')})")
        write_chunk_text(left, lc.get("text", ""))

        for i, sid in enumerate(source_ids):
            rc = rec_by_id.get(sid)
            if i > 0:
                right.add_paragraph()
            if rc is None:
                write_label(right, f"Recursive #{sid} (MISSING)")
                continue
            write_label(right, f"Recursive #{sid} (tokens={rc.get('tokens', '?')})")
            write_chunk_text(right, rc.get("text", ""))

        if idx < len(log_chunks) - 1:
            doc.add_page_break()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing both -chunks.json and -logic-chunks.json files",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write .docx files (default: <input-dir>/reports)",
    )
    ap.add_argument(
        "--doc-id", default=None,
        help="Process only this doc_id (default: all pairs in input-dir)",
    )
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"input-dir does not exist: {input_dir}", file=sys.stderr)
        return 2
    output_dir: Path = args.output_dir or (input_dir / "reports")

    doc_ids = [args.doc_id] if args.doc_id else discover_doc_ids(input_dir)
    if not doc_ids:
        print(f"no *{LOGIC_SUFFIX} files found in {input_dir}", file=sys.stderr)
        return 1

    n_ok = 0
    for doc_id in doc_ids:
        pair = load_pair(input_dir, doc_id)
        if pair is None:
            continue
        rec, log = pair
        out_path = output_dir / f"{doc_id}-chunking-report.docx"
        build_report(rec, log, out_path)
        print(f"[ok] {out_path}")
        n_ok += 1

    print(f"\nwrote {n_ok}/{len(doc_ids)} report(s) to {output_dir}")
    return 0 if n_ok == len(doc_ids) else 1


if __name__ == "__main__":
    sys.exit(main())

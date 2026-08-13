"""Generate full Anima patent/paper document package as PDFs on Desktop."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO = Path(__file__).resolve().parent.parent
PATENT_DIR = REPO / "docs" / "patent"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
if not DESKTOP.is_dir():
    DESKTOP = Path.home() / "Desktop"

TODAY = date.today().isoformat()

# (source markdown relative to PATENT_DIR or REPO, output pdf filename on Desktop)
DOCUMENTS: list[tuple[str, str]] = [
    ("MASTER_PLANS_LIST.md", "Anima_Master_Plans_List.pdf"),
    ("PATENT_FILING_LETTER.md", "Anima_Patent_Filing_Letter.pdf"),
    ("ANNEX_A_INVENTION_DISCLOSURE.md", "Anima_Annex_A_Invention_Disclosure.pdf"),
    ("ANNEX_B_PRIOR_ART.md", "Anima_Annex_B_Prior_Art.pdf"),
    ("PAPER_DRAFT.md", "Anima_Paper_Draft.pdf"),
    ("AUDIT_ISSUES.md", "Anima_Audit_Issues.pdf"),
    ("COUNCIL_SYNTHESIS.md", "Anima_Council_Synthesis.pdf"),
]


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontSize=15,
            spaceAfter=12,
            textColor=colors.HexColor("#1a1a2e"),
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#16213e"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "mono": ParagraphStyle(
            "Mono",
            parent=base["Code"],
            fontSize=8,
            leading=10,
            fontName="Courier",
            spaceAfter=5,
        ),
    }


def _escape(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier' size='8'>\1</font>", text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return text


def _parse_table(lines: list[str]) -> Table | None:
    if len(lines) < 2:
        return None
    rows = []
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.match(r"^[-:\s]+$", c) for c in cells):
            continue
        rows.append([Paragraph(_escape(c), ParagraphStyle("tc", fontSize=8, leading=10)) for c in cells])
    if not rows:
        return None
    col_count = max(len(r) for r in rows)
    width = (6.5 * inch) / col_count
    t = Table(rows, colWidths=[width] * col_count, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def markdown_to_story(md_text: str, styles: dict) -> list:
    story: list = []
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []
    table_buf: list[str] = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            tbl = _parse_table(table_buf)
            if tbl:
                story.append(Spacer(1, 0.05 * inch))
                story.append(tbl)
                story.append(Spacer(1, 0.08 * inch))
            table_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()
            if in_code:
                story.append(Paragraph(_escape("\n".join(code_buf)), styles["mono"]))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if stripped.startswith("|"):
            table_buf.append(line)
            i += 1
            continue
        flush_table()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(_escape(stripped[2:]), styles["title"]))
            i += 1
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(_escape(stripped[3:]), styles["h1"]))
            i += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(_escape(stripped[4:]), styles["h2"]))
            i += 1
            continue
        if stripped.startswith("#### "):
            story.append(Paragraph(_escape(stripped[5:]), styles["h3"]))
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph(f"&bull; {_escape(stripped[2:])}", styles["bullet"]))
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            story.append(Paragraph(_escape(stripped), styles["bullet"]))
            i += 1
            continue

        if stripped.startswith("> "):
            story.append(Paragraph(f"<i>{_escape(stripped[2:])}</i>", styles["body"]))
            i += 1
            continue

        # paragraph (may span if next line is continuation — skip for simplicity)
        story.append(Paragraph(_escape(stripped), styles["body"]))
        i += 1

    flush_table()
    if in_code and code_buf:
        story.append(Paragraph(_escape("\n".join(code_buf)), styles["mono"]))
    return story


USA_PAPER_BLOCKS = [
    ("title", "Anima - USA Research Paper Submission Plan"),
    ("subtitle", f"Siddarth Boggarapu | {TODAY} | github.com/Siddarthb07/Anima"),
    ("h1", "1. Current state"),
    ("body", "Anima v2.0.0 ships: forward hooks, affect probes, guard abstention, layer-disagreement, rolling stability, optional steering, FastAPI + dashboard, multi-judge validity council."),
    ("body", "<b>Blockers:</b> guard AUROC 1.00 is fixture-only (n=52); no live HaluEval/TruthfulQA; no ablations; no semantic-entropy baseline; brain tier is synthetic_minimal."),
    ("h1", "2. Hero model and positioning"),
    ("body", "<b>Hero:</b> TinyLlama-1.1B-Chat (council 94.0, pos-neg gap 0.488). <b>Backup:</b> Qwen2.5-0.5B (91.0, gap 0.267)."),
    ("body", "Frame as systems + methodology contribution. Do NOT claim hallucination detector or brain alignment."),
    ("h1", "3. Required experiments"),
    ("bullet", "E1 (weeks 1-2): Real guard eval on live LM generations - HaluEval + TruthfulQA subsets."),
    ("bullet", "E2 (weeks 3-4): Ablations - disable layer-disagreement, stability, hedging, steering individually."),
    ("bullet", "E3 (weeks 3-4): Semantic entropy + self-consistency (n=5) baselines."),
    ("bullet", "E4 (week 5): MERA comparison on matched task."),
    ("h1", "4. Acceptance probability"),
    ("body", "Workshop now: 20-30%. After E1 + figures: 45-55%. Findings after E1-E4: 55-65%. Main conference reliability: under 15%."),
    ("h1", "5. 12-week timeline"),
    ("body", "W1-2: E1 | W3-4: E2+E3 | W5: E4+figures | W6: polish | W7: patent decision + arXiv if cleared | W8-10: workshop submit | W11-12: HF Space demo."),
]


INDIA_PATENT_BLOCKS = [
    ("title", "Anima and Portfolio - India Patent Filing Plan"),
    ("subtitle", f"Siddarth Boggarapu | {TODAY} | Indian resident"),
    ("h1", "1. Anima core - not fileable in India"),
    ("body", "Public GitHub since 2026-05-17. Section 31 grace does NOT cover open-source release. Verdict: disclosed Anima core cannot be patented in India."),
    ("h1", "2. US still possible"),
    ("body", "US grace until ~2027-05-17. Form 25 FFL may be required before US filing as Indian resident."),
    ("h1", "3. Future India candidates (file BEFORE publish)"),
    ("bullet", "Candidate A - Unified guard fusion: wire layer-disagreement + stability into abstain (35-45% if undisclosed)."),
    ("bullet", "Candidate B - Real ds002345 v3 pipeline (25-35% if undisclosed)."),
    ("bullet", "NeuralVortex FNO on CFD data (40-55% if still private) - BEST portfolio shot."),
    ("h1", "4. Recommended priority"),
    ("body", "P1: Skip India spend on Anima core. P2: NeuralVortex if private. P3: Unified guard if built privately. P4: US provisional for Anima combination only."),
    ("body", "<b>Cost:</b> Rs 25,000-75,000 per India provisional. Rs 15,000-40,000 novelty search."),
]


def _build_blocks(story, blocks, styles):
    for kind, text in blocks:
        if kind == "title":
            story.append(Paragraph(text, styles["title"]))
        elif kind == "subtitle":
            story.append(Paragraph(text, styles["body"]))
            story.append(Spacer(1, 0.1 * inch))
        elif kind == "h1":
            story.append(Spacer(1, 0.06 * inch))
            story.append(Paragraph(text, styles["h1"]))
        elif kind == "bullet":
            story.append(Paragraph(f"&bull; {text}", styles["bullet"]))
        else:
            story.append(Paragraph(text, styles["body"]))


def write_pdf_from_markdown(src: Path, out_name: str) -> Path:
    styles = _styles()
    out = DESKTOP / out_name
    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    md = src.read_text(encoding="utf-8")
    story = markdown_to_story(md, styles)
    doc.build(story)
    return out


def write_pdf_from_blocks(out_name: str, blocks: list[tuple[str, str]]) -> Path:
    styles = _styles()
    out = DESKTOP / out_name
    doc = SimpleDocTemplate(str(out), pagesize=letter, rightMargin=0.7 * inch, leftMargin=0.7 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story: list = []
    _build_blocks(story, blocks, styles)
    doc.build(story)
    return out


def main():
    generated: list[Path] = []

    for src_name, pdf_name in DOCUMENTS:
        src = PATENT_DIR / src_name
        if not src.exists():
            print(f"SKIP missing: {src}")
            continue
        p = write_pdf_from_markdown(src, pdf_name)
        generated.append(p)
        # also copy markdown to desktop
        md_out = DESKTOP / pdf_name.replace(".pdf", ".md")
        md_out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    generated.append(write_pdf_from_blocks("Anima_USA_Research_Paper_Plan.pdf", USA_PAPER_BLOCKS))
    generated.append(write_pdf_from_blocks("Anima_India_Patent_Plan.pdf", INDIA_PATENT_BLOCKS))

    print("Generated on Desktop:")
    for p in generated:
        print(f"  {p}")

    index = DESKTOP / "Anima_Document_Package_Index.txt"
    index.write_text(
        "Anima Document Package\n"
        f"Generated: {TODAY}\n\n"
        + "\n".join(f"- {p.name}" for p in sorted(generated, key=lambda x: x.name))
        + "\n\nStart with: Anima_Master_Plans_List.pdf\n",
        encoding="utf-8",
    )
    print(f"  {index}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import html
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
SOURCE_MD = OUT_DIR / "n8_robust_direct_speed_bo_formulation.md"
OUT_PATH = OUT_DIR / "n8_robust_direct_speed_bo_formulation.docx"


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="260"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="260" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Equation">
    <w:name w:val="Equation"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="180"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:i/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:ind w:left="360"/><w:spacing w:after="120"/></w:pPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:qFormat/>
    <w:tblPr>
      <w:tblBorders>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="80" w:type="dxa"/>
        <w:left w:w="80" w:type="dxa"/>
        <w:bottom w:w="80" w:type="dxa"/>
        <w:right w:w="80" w:type="dxa"/>
      </w:tblCellMar>
    </w:tblPr>
  </w:style>
</w:styles>
"""

APP = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Word</Application>
</Properties>
"""


def _clean_inline_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


def _paragraph_xml(style: str, text: str) -> str:
    escaped = html.escape(_clean_inline_markdown(text))
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>'
    )


def _equation_xml(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return _paragraph_xml("Equation", text)


def _table_xml(rows: list[list[str]]) -> str:
    table_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row:
            shading = '<w:shd w:fill="EDEDED"/>' if row_index == 0 else ""
            text = html.escape(_clean_inline_markdown(cell.strip()))
            cells.append(
                "<w:tc>"
                f"<w:tcPr>{shading}</w:tcPr>"
                "<w:p><w:pPr><w:pStyle w:val=\"Normal\"/></w:pPr>"
                f"<w:r><w:t xml:space=\"preserve\">{text}</w:t></w:r>"
                "</w:p></w:tc>"
            )
        table_rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/></w:tblPr>'
        + "".join(table_rows)
        + "</w:tbl>"
    )


def _parse_markdown_blocks(markdown_text: str) -> list[tuple[str, str | list[list[str]] | list[str]]]:
    blocks: list[tuple[str, str | list[list[str]] | list[str]]] = []
    paragraph_lines: list[str] = []
    equation_lines: list[str] = []
    table_lines: list[str] = []
    in_equation = False

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines)
            blocks.append(("Paragraph", text))
            paragraph_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        rows: list[list[str]] = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            rows.append(cells)
        if rows:
            blocks.append(("Table", rows))
        table_lines = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.strip() == "$$":
            flush_paragraph()
            flush_table()
            if in_equation:
                blocks.append(("Equation", equation_lines))
                equation_lines = []
                in_equation = False
            else:
                in_equation = True
            continue

        if in_equation:
            equation_lines.append(line)
            continue

        if line.startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue

        flush_table()

        if line.startswith("# "):
            flush_paragraph()
            blocks.append(("Title", line[2:].strip()))
            continue

        if line.startswith("## "):
            flush_paragraph()
            blocks.append(("Heading", line[3:].strip()))
            continue

        if not line.strip():
            flush_paragraph()
            continue

        if re.match(r"^\d+\.\s+", line.strip()):
            flush_paragraph()
            blocks.append(("List", line.strip()))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    flush_table()
    return blocks


def _blocks_to_document_xml(blocks: list[tuple[str, str | list[list[str]] | list[str]]]) -> str:
    body_parts: list[str] = []
    for kind, content in blocks:
        if kind == "Title":
            body_parts.append(_paragraph_xml("Title", str(content)))
        elif kind == "Heading":
            body_parts.append(_paragraph_xml("Heading1", str(content)))
        elif kind == "Equation":
            body_parts.append(_equation_xml(list(content)))  # type: ignore[arg-type]
        elif kind == "Table":
            body_parts.append(_table_xml(list(content)))  # type: ignore[arg-type]
        elif kind == "List":
            body_parts.append(_paragraph_xml("ListParagraph", str(content)))
        else:
            body_parts.append(_paragraph_xml("Normal", str(content)))

    body = "\n".join(body_parts)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1152" w:bottom="1440" w:left="1152"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def _core_xml() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Robust n8 Bayesian optimization with direct-speed objective alignment</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def main() -> None:
    markdown_text = SOURCE_MD.read_text(encoding="utf-8")
    blocks = _parse_markdown_blocks(markdown_text)
    document_xml = _blocks_to_document_xml(blocks)
    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", RELS)
        archive.writestr("word/_rels/document.xml.rels", DOC_RELS)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", STYLES)
        archive.writestr("docProps/core.xml", _core_xml())
        archive.writestr("docProps/app.xml", APP)
    print(OUT_PATH)


if __name__ == "__main__":
    main()

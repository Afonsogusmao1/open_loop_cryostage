from __future__ import annotations

import html
import zipfile
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "BO3_monotone_admissible_parameterization.docx"


SECTIONS = [
    ("Title", "Bayesian Optimization with Monotone Admissible Parameterization"),
    (
        "Paragraph",
        "To improve the robustness of the open-loop cryostage optimization, the Bayesian "
        "optimization (BO) was reformulated so that the optimizer no longer samples the "
        "physical cryostage reference temperatures directly. Instead, the BO operates in a "
        "normalized unit hypercube and each proposed point is mapped into a physically "
        "admissible temperature trajectory before the expensive FEM simulation is evaluated.",
    ),
    ("Heading", "Control Parameterization"),
    (
        "Paragraph",
        "Let N be the number of temperature control knots and let theta = "
        "(theta_1, theta_2, ..., theta_N) denote the cryostage reference temperatures "
        "prescribed at fixed knot times 0 = t_1 < t_2 < ... < t_N = t_f. The corresponding "
        "continuous reference trajectory T_ref(t) is obtained by piecewise-linear "
        "interpolation of the pairs (t_i, theta_i).",
    ),
    ("Equation", "T_ref(t) = Interp[(t_i, theta_i)],   i = 1, ..., N."),
    (
        "Paragraph",
        "In the previous BO formulation, the optimizer searched directly over theta subject "
        "to node-wise bounds theta_i in [L_i, U_i]. In the new formulation, the optimizer "
        "samples normalized variables",
    ),
    ("Equation", "u = (u_1, u_2, ..., u_N),   u_i in [0, 1]."),
    (
        "Paragraph",
        "These variables are then transformed into physical temperatures theta_i while "
        "enforcing both the individual bounds and a non-increasing temperature trajectory,",
    ),
    ("Equation", "L_i <= theta_i <= U_i,   theta_1 >= theta_2 >= ... >= theta_N."),
    ("Heading", "Recursive Mapping from the Unit Hypercube"),
    (
        "Paragraph",
        "The transformation is performed recursively from the last knot to the first. "
        "For the final knot,",
    ),
    ("Equation", "theta_N = L_N + u_N (U_N - L_N)."),
    (
        "Paragraph",
        "For the remaining knots, i = N-1, ..., 1, the lower bound is made dependent on the "
        "next temperature value so that monotonicity is guaranteed by construction:",
    ),
    (
        "Equation",
        "theta_i = max(L_i, theta_{i+1}) + u_i [U_i - max(L_i, theta_{i+1})].",
    ),
    (
        "Paragraph",
        "Therefore, every candidate sampled by the BO in the normalized space is mapped to "
        "a trajectory that satisfies theta_i >= theta_{i+1} and remains inside the "
        "prescribed node-wise bounds. The optimization problem can then be written as",
    ),
    ("Equation", "u* = arg min_{u in [0,1]^N} J(theta(u)),"),
    (
        "Paragraph",
        "where J is the original front-tracking objective. Importantly, the physical "
        "objective function was not changed; only the search-space parameterization was "
        "modified.",
    ),
    ("Heading", "Admissibility-Aware Initialization"),
    (
        "Paragraph",
        "The initial design points of the BO were also modified. Rather than drawing the "
        "initial candidates uniformly from the full physical temperature box, the canonical "
        "seed trajectory theta0 is first mapped into the normalized space. Local "
        "perturbations are then generated around this normalized seed point. Each candidate "
        "is transformed back into a physical theta trajectory and passed through the "
        "inexpensive admissibility precheck before any FEM evaluation is launched.",
    ),
    (
        "Equation",
        "u0 = Phi^{-1}(theta0),   u_init = clip(u0 + epsilon, 0, 1),   epsilon ~ local perturbation.",
    ),
    (
        "Paragraph",
        "Only candidates that pass the admissibility precheck are evaluated with the "
        "expensive FEM model. If the local sampling stage cannot provide enough feasible "
        "initial candidates, a global fallback in the normalized space is used, also with "
        "the same precheck. This avoids spending computational budget on candidates that "
        "are already known to violate the admissibility constraints.",
    ),
    ("Heading", "Current BO3 Configuration"),
    (
        "Paragraph",
        "The current BO3 configuration keeps the front-tracking objective fixed and uses "
        "the following optimizer settings: expected improvement acquisition, xi = 0.01, "
        "seed_with_theta0 = true, init_points = 2, n_iter = 30, parameterization_kind = "
        "monotone_unit_box, and init_strategy = feasible_local. Therefore, each BO run "
        "contains one explicit evaluation of theta0, two admissibility-aware initial "
        "points, and thirty acquisition-guided BO iterations, for a maximum of thirty-three "
        "FEM evaluations per run.",
    ),
    ("Heading", "Scientific Rationale"),
    (
        "Paragraph",
        "This reparameterization does not impose a new physical model; it restricts the BO "
        "search to a physically motivated subset of candidate cryostage trajectories. Since "
        "the desired control action is expected to become progressively colder as the "
        "freezing front moves upward and the frozen layer increases the thermal resistance, "
        "non-increasing reference trajectories are a reasonable modeling choice for this "
        "phase of the optimization. The purpose is to improve search efficiency and "
        "seed-to-seed robustness before drawing conclusions about whether increasing the "
        "number of control knots is necessary.",
    ),
    ("Heading", "Short Manuscript Text"),
    (
        "Paragraph",
        "To improve the efficiency and robustness of the Bayesian optimization, the "
        "cryostage reference temperatures were reparameterized in a normalized unit "
        "hypercube. Instead of sampling the physical knot temperatures directly, the "
        "optimizer samples u_i in [0,1], which are recursively mapped into physical "
        "temperatures theta_i while enforcing node-wise bounds and a non-increasing "
        "temperature trajectory. This guarantees that every proposed candidate satisfies "
        "L_i <= theta_i <= U_i and theta_i >= theta_{i+1} before the expensive FEM "
        "evaluation is performed. The original front-tracking objective is kept unchanged; "
        "only the admissible search space and the initialization strategy are modified.",
    ),
]


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
    <w:pPr><w:spacing w:after="240"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Equation">
    <w:name w:val="Equation"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="160"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:i/><w:sz w:val="22"/></w:rPr>
  </w:style>
</w:styles>
"""


def paragraph_xml(style: str, text: str) -> str:
    style_id = {"Title": "Title", "Heading": "Heading1", "Equation": "Equation"}.get(style, "Normal")
    escaped = html.escape(text)
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style_id}"/></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>'
    )


def build_document_xml() -> str:
    body = "".join(paragraph_xml(kind, text) for kind, text in SECTIONS)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def build_core_xml() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Bayesian Optimization with Monotone Admissible Parameterization</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


APP = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Word</Application>
</Properties>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", RELS)
        archive.writestr("word/_rels/document.xml.rels", DOC_RELS)
        archive.writestr("word/document.xml", build_document_xml())
        archive.writestr("word/styles.xml", STYLES)
        archive.writestr("docProps/core.xml", build_core_xml())
        archive.writestr("docProps/app.xml", APP)
    print(OUT_PATH)


if __name__ == "__main__":
    main()

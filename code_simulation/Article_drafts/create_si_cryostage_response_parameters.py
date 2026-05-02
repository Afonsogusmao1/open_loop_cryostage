from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from code_simulation.simulation.cryostage_model import (
    DEFAULT_CRYOSTAGE_PARAMS,
    default_characterization_run_paths,
    fit_default_cryostage_params,
    load_characterization_run,
    simulate_characterization_run,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

MD_PATH = BASE_DIR / "SI_cryostage_first_order_response_parameters.md"
DOCX_PATH = BASE_DIR / "SI_cryostage_first_order_response_parameters.docx"
DOCX_FALLBACK_PATH = BASE_DIR / "SI_cryostage_active_first_order_response_parameters.docx"
PARAM_TABLE_CSV = BASE_DIR / "SI_cryostage_first_order_response_parameter_table.csv"
FIT_SUMMARY_CSV = BASE_DIR / "SI_cryostage_first_order_response_fit_summary.csv"


def _target_from_path(path: Path) -> float:
    parent = path.parent.name
    if not parent.startswith("characterization_min"):
        return float("nan")
    return -float(parent.removeprefix("characterization_min"))


def _run_recurrence_fit_residuals(runs, params) -> np.ndarray:
    residuals: list[np.ndarray] = []
    coeffs = np.asarray([params.gain, params.offset_C], dtype=np.float64)
    for run in runs:
        dt_s = np.diff(run.time_s)
        alpha = np.exp(-dt_s / params.tau_s)
        target = run.T_plate_C[1:] - alpha * run.T_plate_C[:-1]
        phi = np.column_stack(((1.0 - alpha) * run.T_ref_C[:-1], 1.0 - alpha))
        residuals.append(phi @ coeffs - target)
    if not residuals:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(residuals)


def _global_recursive_rmse(runs, params) -> float:
    sse = 0.0
    count = 0
    for run in runs:
        predicted = simulate_characterization_run(run, params)
        err = predicted - run.T_plate_C
        sse += float(np.dot(err, err))
        count += int(err.size)
    return math.sqrt(sse / count)


def _per_target_recursive_rmse(paths, runs, params) -> dict[float, float]:
    by_target: dict[float, list[float]] = defaultdict(list)
    for path, run in zip(paths, runs, strict=True):
        predicted = simulate_characterization_run(run, params)
        err = predicted - run.T_plate_C
        by_target[_target_from_path(path)].append(float(np.sqrt(np.mean(err * err))))
    return {target: float(np.mean(values)) for target, values in sorted(by_target.items())}


def _collect_metrics() -> dict[str, object]:
    paths = list(default_characterization_run_paths(PROJECT_ROOT))
    active_power_threshold = 1.0
    runs = [
        load_characterization_run(path, active_power_threshold=active_power_threshold)
        for path in paths
    ]
    fitted = fit_default_cryostage_params(PROJECT_ROOT)

    if (
        not math.isclose(fitted.tau_s, DEFAULT_CRYOSTAGE_PARAMS.tau_s, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(fitted.gain, DEFAULT_CRYOSTAGE_PARAMS.gain, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(fitted.offset_C, DEFAULT_CRYOSTAGE_PARAMS.offset_C, rel_tol=0.0, abs_tol=1e-6)
        or not np.allclose(
            fitted.reference_temperatures_C,
            DEFAULT_CRYOSTAGE_PARAMS.reference_temperatures_C,
            rtol=0.0,
            atol=1e-6,
        )
        or not np.allclose(
            fitted.response_tau_s,
            DEFAULT_CRYOSTAGE_PARAMS.response_tau_s,
            rtol=0.0,
            atol=1e-6,
        )
        or not np.allclose(
            fitted.steady_plate_C,
            DEFAULT_CRYOSTAGE_PARAMS.steady_plate_C,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise RuntimeError("Recomputed cryostage parameters do not match DEFAULT_CRYOSTAGE_PARAMS")

    all_dt = np.concatenate([np.diff(run.time_s) for run in runs])
    targets = sorted({_target_from_path(path) for path in paths})
    per_target_counts = {
        target: sum(1 for path in paths if math.isclose(_target_from_path(path), target))
        for target in targets
    }
    active_duration_by_target: dict[float, list[float]] = defaultdict(list)
    tau_by_target = {
        float(reference): float(tau_s)
        for reference, tau_s in zip(
            fitted.reference_temperatures_C,
            fitted.response_tau_s,
            strict=True,
        )
    }
    duration_tau_ratios: list[float] = []
    for path, run in zip(paths, runs, strict=True):
        target = _target_from_path(path)
        duration_s = float(run.time_s[-1] - run.time_s[0])
        active_duration_by_target[target].append(duration_s)
        duration_tau_ratios.append(duration_s / tau_by_target[target])

    return {
        "paths": paths,
        "runs": runs,
        "params": fitted,
        "targets": targets,
        "active_power_threshold": active_power_threshold,
        "per_target_counts": per_target_counts,
        "n_files": len(paths),
        "n_samples": sum(int(run.time_s.size) for run in runs),
        "n_pairs": sum(max(int(run.time_s.size) - 1, 0) for run in runs),
        "median_dt_s": float(np.median(all_dt)),
        "active_duration_min_s": min(min(values) for values in active_duration_by_target.values()),
        "active_duration_max_s": max(max(values) for values in active_duration_by_target.values()),
        "active_duration_by_target_s": {
            target: (min(values), max(values))
            for target, values in sorted(active_duration_by_target.items())
        },
        "active_duration_tau_ratio_min": min(duration_tau_ratios),
        "active_duration_tau_ratio_max": max(duration_tau_ratios),
        "recursive_rmse_C": _global_recursive_rmse(runs, fitted),
        "per_target_recursive_rmse_C": _per_target_recursive_rmse(paths, runs, fitted),
    }


def _parameter_rows(metrics: dict[str, object]) -> list[dict[str, str]]:
    params = metrics["params"]
    reference_text = ", ".join(f"{value:.0f}" for value in params.reference_temperatures_C)
    tau_lookup_text = ", ".join(f"{value:.3f}" for value in params.response_tau_s)
    steady_lookup_text = ", ".join(f"{value:.3f}" for value in params.steady_plate_C)
    return [
        {
            "quantity": "Representative plate-temperature response time constant",
            "symbol": "median tau",
            "code_field": "tau_s",
            "value": f"{params.tau_s:.6f}",
            "unit": "s",
            "estimation": "Median of the setpoint-dependent response-time lookup.",
        },
        {
            "quantity": "Linear summary gain for the steady-state lookup",
            "symbol": "g",
            "code_field": "gain",
            "value": f"{params.gain:.6f}",
            "unit": "dimensionless",
            "estimation": "Least-squares linear summary of the fitted T_plate,ss(T_ref) lookup.",
        },
        {
            "quantity": "Linear summary offset for the steady-state lookup",
            "symbol": "b",
            "code_field": "offset_C",
            "value": f"{params.offset_C:.6f}",
            "unit": "deg C",
            "estimation": "Least-squares linear summary of the fitted T_plate,ss(T_ref) lookup.",
        },
        {
            "quantity": "Characterized reference temperatures",
            "symbol": "T_ref",
            "code_field": "reference_temperatures_C",
            "value": reference_text,
            "unit": "deg C",
            "estimation": "Nominal setpoints of the active dry-cooling characterization assays.",
        },
        {
            "quantity": "Setpoint-dependent response time constants",
            "symbol": "tau(T_ref)",
            "code_field": "response_tau_s",
            "value": tau_lookup_text,
            "unit": "s",
            "estimation": "Independent active step-response trajectory fit at each characterized setpoint.",
        },
        {
            "quantity": "Setpoint-dependent steady plate temperatures",
            "symbol": "T_plate,ss(T_ref)",
            "code_field": "steady_plate_C",
            "value": steady_lookup_text,
            "unit": "deg C",
            "estimation": "Independent active step-response trajectory fit at each characterized setpoint.",
        },
    ]


def _fit_summary_rows(metrics: dict[str, object]) -> list[dict[str, str]]:
    target_text = ", ".join(f"{target:.0f}" for target in metrics["targets"])
    replicate_text = ", ".join(
        f"{target:.0f} C: {count}" for target, count in metrics["per_target_counts"].items()
    )
    per_target_rmse = metrics["per_target_recursive_rmse_C"]
    per_target_rmse_text = ", ".join(
        f"{target:.0f} C: {rmse:.3f} C" for target, rmse in per_target_rmse.items()
    )
    duration_by_target = metrics["active_duration_by_target_s"]
    duration_by_target_text = ", ".join(
        f"{target:.0f} C: {duration_min:.1f}-{duration_max:.1f} s"
        for target, (duration_min, duration_max) in duration_by_target.items()
    )
    return [
        {"item": "Characterization files used", "value": str(metrics["n_files"])},
        {"item": "Nominal reference targets", "value": f"{target_text} deg C"},
        {"item": "Replicates per nominal target", "value": replicate_text},
        {
            "item": "Active cooling threshold",
            "value": f"First telemetry row with power > {metrics['active_power_threshold']:.1f}",
        },
        {"item": "Active telemetry samples retained", "value": str(metrics["n_samples"])},
        {"item": "Adjacent active sample pairs", "value": str(metrics["n_pairs"])},
        {"item": "Median telemetry sampling interval", "value": f"{metrics['median_dt_s']:.3f} s"},
        {
            "item": "Active cooling duration after trimming",
            "value": (
                f"{metrics['active_duration_min_s']:.1f}-{metrics['active_duration_max_s']:.1f} s "
                f"across runs; by target: {duration_by_target_text}"
            ),
        },
        {
            "item": "Active duration relative to fitted response time",
            "value": (
                f"{metrics['active_duration_tau_ratio_min']:.1f}-"
                f"{metrics['active_duration_tau_ratio_max']:.1f} fitted time constants"
            ),
        },
        {"item": "Time-constant search interval", "value": "10-250 s for each characterized setpoint"},
        {
            "item": "Fitting criterion",
            "value": "Minimum pooled active step-response trajectory RMSE at each setpoint",
        },
        {
            "item": "Pooled recursive trajectory RMSE on active telemetry",
            "value": f"{metrics['recursive_rmse_C']:.6f} deg C",
        },
        {
            "item": "Mean recursive RMSE by nominal target on active telemetry",
            "value": per_target_rmse_text,
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_markdown(metrics: dict[str, object], parameter_rows: list[dict[str, str]], fit_rows: list[dict[str, str]]) -> None:
    paths = metrics["paths"]
    parameter_table = _markdown_table(
        ["Quantity", "Symbol", "Code field", "Value", "Unit", "Estimation"],
        [
            [
                row["quantity"],
                row["symbol"],
                row["code_field"],
                row["value"],
                row["unit"],
                row["estimation"],
            ]
            for row in parameter_rows
        ],
    )
    fit_table = _markdown_table(
        ["Item", "Value"],
        [[row["item"], row["value"]] for row in fit_rows],
    )
    file_list = "\n".join(f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`" for path in paths)

    text = f"""# Supplementary note: first-order cryostage response parameters

The plate-temperature response parameters used in the open-loop trajectory evaluations were obtained from the cryostage characterization telemetry, rather than adjusted during the trajectory optimization. The calibration uses the files matching `data/characterization_cryostage/characterization_min*/cryostage_characterization_min*.csv`. In the implementation, comment-prefixed metadata lines are skipped, only rows with `row_type == telemetry` are retained, and the columns `panel_t_s`, `set`, `T_cal`, and `power` are interpreted as sample time, requested reference temperature, calibrated plate temperature, and controller output, respectively. Each run is trimmed to the active-cooling interval beginning at the first telemetry row with `power > 1.0`; the samples are then sorted by time, duplicate time stamps are removed, and the time axis is shifted so that the first active sample occurs at `t = 0`.

The fitted response is the same first-order recurrence used during the freezing simulations, but the response time and steady plate temperature are functions of the requested reference temperature:

```text
T_plate,i = alpha_i T_plate,i-1 + (1 - alpha_i) T_plate,ss(T_ref,i-1),
alpha_i = exp(-Delta t_i / tau(T_ref,i-1)).
```

For each characterized setpoint, the active step-response trajectory is fitted as `T_plate(t) = h(t) T_plate(0) + (1 - h(t)) T_plate,ss`, with `h(t) = exp(-t/tau)`. The code evaluates `tau` on a logarithmically spaced grid from 10 to 250 s and estimates `T_plate,ss` by least squares over the full active trajectory of the three replicate runs at that setpoint. The active cooling intervals lasted 454.3-795.5 s after trimming, corresponding to 6.0-8.0 fitted time constants for their respective setpoints; therefore `T_plate,ss` is treated as a fitted asymptotic steady-state term rather than a temperature read after an arbitrary 100-300 s hold. During freezing simulations, `tau(T_ref)` and `T_plate,ss(T_ref)` are obtained by linear interpolation of the characterized lookup values, with linear extrapolation only just outside the characterized interval.

This calibration should be interpreted as a low-order empirical mapping from requested reference temperature to measured plate temperature under the characterization conditions. It does not use freezing-front information and does not define a feedback law for freezing-front control. In the open-loop simulations, it is used only to generate the time-dependent plate-temperature input that is applied as the lower boundary condition in the freezing solver.

## Table SX. Calibrated first-order response parameters

{parameter_table}

## Fit and data summary

{fit_table}

## Characterization files included by the default loader

{file_list}
"""
    MD_PATH.write_text(text, encoding="utf-8")


def _run(text: str, *, bold: bool = False, font: str | None = None) -> str:
    props: list[str] = []
    if bold:
        props.append("<w:b/>")
    if font:
        props.append(
            f'<w:rFonts w:ascii="{escape(font)}" w:hAnsi="{escape(font)}" w:cs="{escape(font)}"/>'
        )
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _paragraph(text: str, *, style: str | None = None, after: int = 160, font: str | None = None) -> str:
    ppr = [f'<w:spacing w:after="{after}"/>']
    if style:
        ppr.insert(0, f'<w:pStyle w:val="{style}"/>')
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{_run(text, font=font)}</w:p>"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(text: str, *, header: bool = False) -> str:
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p>{_run(text, bold=header)}</w:p></w:tc>"
        )

    all_rows = [headers] + rows
    row_xml = []
    for index, row in enumerate(all_rows):
        row_xml.append("<w:tr>" + "".join(cell(item, header=index == 0) for item in row) + "</w:tr>")
    return (
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "</w:tblBorders></w:tblPr>"
        + "".join(row_xml)
        + "</w:tbl>"
    )


def _content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""


def _rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def _document_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


def _styles() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:line="276" w:lineRule="auto" w:after="160"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="240" w:after="220"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="200" w:after="180"/><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="24"/></w:rPr>
  </w:style>
</w:styles>
"""


def _write_docx_archive(path: Path, document: str) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", _content_types())
        docx.writestr("_rels/.rels", _rels())
        docx.writestr("word/_rels/document.xml.rels", _document_rels())
        docx.writestr("word/styles.xml", _styles())
        docx.writestr("word/document.xml", document)


def _write_docx(parameter_rows: list[dict[str, str]], fit_rows: list[dict[str, str]]) -> Path:
    parameter_table = _table(
        ["Quantity", "Symbol", "Code field", "Value", "Unit", "Estimation"],
        [
            [
                row["quantity"],
                row["symbol"],
                row["code_field"],
                row["value"],
                row["unit"],
                row["estimation"],
            ]
            for row in parameter_rows
        ],
    )
    fit_table = _table(["Item", "Value"], [[row["item"], row["value"]] for row in fit_rows])
    body = "".join(
        [
            _paragraph("Supplementary note: first-order cryostage response parameters", style="Heading1"),
            _paragraph(
                "The plate-temperature response parameters used in the open-loop trajectory evaluations were obtained from the cryostage characterization telemetry, rather than adjusted during the trajectory optimization. The calibration uses the files matching data/characterization_cryostage/characterization_min*/cryostage_characterization_min*.csv. In the implementation, comment-prefixed metadata lines are skipped, only rows with row_type == telemetry are retained, and the columns panel_t_s, set, T_cal, and power are interpreted as sample time, requested reference temperature, calibrated plate temperature, and controller output, respectively. Each run is trimmed to the active-cooling interval beginning at the first telemetry row with power > 1.0; the samples are then sorted by time, duplicate time stamps are removed, and the time axis is shifted so that the first active sample occurs at t = 0."
            ),
            _paragraph(
                "The fitted response is the same first-order recurrence used during the freezing simulations, but the response time and steady plate temperature are functions of the requested reference temperature:"
            ),
            _paragraph(
                "T_plate,i = alpha_i T_plate,i-1 + (1 - alpha_i) T_plate,ss(T_ref,i-1),    alpha_i = exp(-Delta t_i / tau(T_ref,i-1)).",
                font="Cambria Math",
            ),
            _paragraph(
                "For each characterized setpoint, the active step-response trajectory is fitted as T_plate(t) = h(t) T_plate(0) + (1 - h(t)) T_plate,ss, with h(t) = exp(-t/tau). The code evaluates tau on a logarithmically spaced grid from 10 to 250 s and estimates T_plate,ss by least squares over the full active trajectory of the three replicate runs at that setpoint. The active cooling intervals lasted 454.3-795.5 s after trimming, corresponding to 6.0-8.0 fitted time constants for their respective setpoints; therefore T_plate,ss is treated as a fitted asymptotic steady-state term rather than a temperature read after an arbitrary 100-300 s hold. During freezing simulations, tau(T_ref) and T_plate,ss(T_ref) are obtained by linear interpolation of the characterized lookup values, with linear extrapolation only just outside the characterized interval."
            ),
            _paragraph(
                "This calibration should be interpreted as a low-order empirical mapping from requested reference temperature to measured plate temperature under the characterization conditions. It does not use freezing-front information and does not define a feedback law for freezing-front control. In the open-loop simulations, it is used only to generate the time-dependent plate-temperature input that is applied as the lower boundary condition in the freezing solver."
            ),
            _paragraph("Table SX. Calibrated first-order response parameters", style="Heading2"),
            parameter_table,
            _paragraph("Fit and data summary", style="Heading2"),
            fit_table,
            "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/></w:sectPr>",
        ]
    )
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}</w:body>
</w:document>
"""
    try:
        _write_docx_archive(DOCX_PATH, document)
        return DOCX_PATH
    except PermissionError:
        _write_docx_archive(DOCX_FALLBACK_PATH, document)
        return DOCX_FALLBACK_PATH


def main() -> None:
    metrics = _collect_metrics()
    parameter_rows = _parameter_rows(metrics)
    fit_rows = _fit_summary_rows(metrics)
    _write_csv(PARAM_TABLE_CSV, parameter_rows)
    _write_csv(FIT_SUMMARY_CSV, fit_rows)
    _write_markdown(metrics, parameter_rows, fit_rows)
    docx_path = _write_docx(parameter_rows, fit_rows)

    print(MD_PATH)
    print(PARAM_TABLE_CSV)
    print(FIT_SUMMARY_CSV)
    print(docx_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Desktop panel for cryostage characterization experiments."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_BOOTSTRAP_ENV = "CRYOSTAGE_PANEL_BOOTSTRAPPED"
_TK_PROBE = "\n".join(
    [
        "import tkinter as tk",
        "import matplotlib",
        "matplotlib.use('TkAgg')",
        "from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg",
        "import serial",
        "root = tk.Tk()",
        "root.withdraw()",
        "root.update_idletasks()",
        "root.destroy()",
    ]
)


def _unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved).lower()
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _python_candidates() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    candidates = [
        Path(sys.executable),
        script_dir / ".venv" / "Scripts" / "python.exe",
        project_dir / ".venv" / "Scripts" / "python.exe",
        project_dir.parent / ".venv" / "Scripts" / "python.exe",
    ]

    local_app_data = os.environ.get("LocalAppData")
    if local_app_data:
        candidates.extend(sorted((Path(local_app_data) / "Programs" / "Python").glob("Python*/python.exe"), reverse=True))

    return _unique_existing_paths(candidates)


def _python_home(python_exe: Path) -> Path:
    try:
        return python_exe.resolve().parent
    except OSError:
        return python_exe.parent


def _tk_env_candidates(python_exe: Path) -> list[dict[str, str]]:
    base_env = os.environ.copy()
    envs: list[dict[str, str]] = [base_env]
    seen: set[tuple[str, str]] = set()

    def add_env(tcl_dir: Path, tk_dir: Path) -> None:
        if not tcl_dir.is_dir() or not tk_dir.is_dir():
            return
        key = (str(tcl_dir).lower(), str(tk_dir).lower())
        if key in seen:
            return
        seen.add(key)
        env = base_env.copy()
        env["TCL_LIBRARY"] = str(tcl_dir)
        env["TK_LIBRARY"] = str(tk_dir)
        envs.append(env)

    python_home = _python_home(python_exe)
    add_env(python_home / "tcl" / "tcl8.6", python_home / "tcl" / "tk8.6")

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    add_env(program_files / "Git" / "mingw64" / "lib" / "tcl8.6", program_files / "Git" / "mingw64" / "lib" / "tk8.6")

    return envs


def _probe_python(python_exe: Path, env: dict[str, str]) -> bool:
    try:
        result = subprocess.run(
            [str(python_exe), "-c", _TK_PROBE],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _apply_gui_env(env: dict[str, str]) -> None:
    for key in ("TCL_LIBRARY", "TK_LIBRARY"):
        value = env.get(key)
        if value:
            os.environ[key] = value


def _ensure_gui_runtime() -> None:
    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        return

    current_python = Path(sys.executable)
    for env in _tk_env_candidates(current_python):
        if _probe_python(current_python, env):
            _apply_gui_env(env)
            os.environ[_BOOTSTRAP_ENV] = "1"
            return

    current_key = str(current_python.resolve()).lower()
    script_path = str(Path(__file__).resolve())
    for python_exe in _python_candidates():
        try:
            python_key = str(python_exe.resolve()).lower()
        except OSError:
            python_key = str(python_exe).lower()
        if python_key == current_key:
            continue

        for env in _tk_env_candidates(python_exe):
            if not _probe_python(python_exe, env):
                continue

            child_env = env.copy()
            child_env[_BOOTSTRAP_ENV] = "1"
            print(f"[INFO] Relaunching panel with GUI-capable Python: {python_exe}", file=sys.stderr)
            raise SystemExit(subprocess.call([str(python_exe), script_path, *sys.argv[1:]], env=child_env))

    print(
        "ERROR: Could not find a Python interpreter that can start the Tk desktop panel.\n"
        "Try installing/running a normal CPython interpreter with Tk support instead of PlatformIO's penv.",
        file=sys.stderr,
    )
    raise SystemExit(1)


_ensure_gui_runtime()

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import serial
from serial.tools import list_ports
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


CSV_COLUMNS = [
    "row_type",
    "host_time_iso",
    "panel_t_s",
    "ts",
    "event",
    "event_detail",
    "set",
    "pid",
    "armed",
    "T_cal",
    "T_raw",
    "power",
    "T3",
    "T7",
    "T12",
    "Tamb",
    "char_en",
    "char_state",
    "char_step",
    "char_total",
    "char_target",
    "char_time_s",
    "stable",
    "timeout",
    "char_in_band",
    "char_filtered",
    "char_slope",
    "fault",
    "fault_msg",
    "fw",
]


def list_serial_port_names() -> list[str]:
    return [port.device for port in list_ports.comports()]


def coerce_value(value: str) -> Any:
    lower = value.lower()
    if lower in {"nan", "inf", "+inf", "-inf"}:
        try:
            return float(value)
        except ValueError:
            return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def parse_telemetry_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line.startswith(">"):
        return None
    payload = line[1:].strip()
    if ":" not in payload:
        return None

    out: Dict[str, Any] = {}
    for part in payload.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        out[key.strip()] = coerce_value(value.strip())
    return out


def parse_pipe_kv_line(line: str, prefix: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line.startswith(prefix):
        return None

    payload = line[len(prefix) :].strip()
    out: Dict[str, Any] = {}
    for chunk in payload.split("|"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[key.strip()] = coerce_value(value.strip())
    return out


def parse_charseq_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line.startswith("[CHARSEQ]"):
        return None
    return line[len("[CHARSEQ]") :].strip()


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def format_number(value: Any, digits: int = 2, fallback: str = "--") -> str:
    x = as_float(value)
    if not math.isfinite(x):
        return fallback
    return f"{x:.{digits}f}"


def build_csv_row(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    row = {column: "" for column in CSV_COLUMNS}
    row["host_time_iso"] = dt.datetime.now().isoformat(timespec="seconds")
    for column in CSV_COLUMNS:
        if column in snapshot:
            row[column] = snapshot[column]
    return row


class CsvLogger:
    def __init__(self) -> None:
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self.path: Optional[Path] = None
        self.recording = False
        self._lock = threading.RLock()

    def start(self, path: Path, metadata: Dict[str, Any]) -> None:
        with self._lock:
            self.stop()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", newline="", encoding="utf-8")
            self.path = path
            self.recording = True

            for key, value in metadata.items():
                self._file.write(f"# {key}: {value}\n")
            self._file.write("#\n")

            self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
            self._writer.writeheader()
            self._file.flush()

    def stop(self) -> None:
        with self._lock:
            self.recording = False
            self.path = None
            if self._file is not None:
                self._file.flush()
                self._file.close()
            self._file = None
            self._writer = None

    def write_row(self, row: Dict[str, Any]) -> None:
        with self._lock:
            if not self.recording or self._writer is None or self._file is None:
                return
            clean_row = {column: row.get(column, "") for column in CSV_COLUMNS}
            self._writer.writerow(clean_row)
            self._file.flush()

    def write_event(self, event: str, detail: str, snapshot: Dict[str, Any]) -> None:
        row = build_csv_row(snapshot)
        row["row_type"] = "event"
        row["event"] = event
        row["event_detail"] = detail
        self.write_row(row)

    def write_telemetry(self, snapshot: Dict[str, Any]) -> None:
        row = build_csv_row(snapshot)
        row["row_type"] = "telemetry"
        self.write_row(row)


class CryostagePanel:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.root.title("Cryostage Characterization Panel")
        self.root.geometry(self._initial_geometry())

        self.lock = threading.RLock()
        self.logger = CsvLogger()

        self.serial_port: Optional[serial.Serial] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.stop_reader = threading.Event()
        self.tx_lock = threading.RLock()

        self.latest: Dict[str, Any] = {}
        self.last_snapshot: Dict[str, Any] = {}
        self.pending_error: Optional[str] = None
        self.log_messages: queue.Queue[str] = queue.Queue()

        self.right_panel_canvas: Optional[tk.Canvas] = None
        self.right_panel_frame: Optional[ttk.Frame] = None
        self.right_panel_window: Optional[int] = None

        self.panel_t0_ms: Optional[float] = None
        self.last_ts_ms: Optional[float] = None

        max_points = max(600, int(args.buffer_sec * 10))
        self.t_buf = deque(maxlen=max_points)
        self.plate_buf = deque(maxlen=max_points)
        self.set_buf = deque(maxlen=max_points)
        self.t3_buf = deque(maxlen=max_points)
        self.t7_buf = deque(maxlen=max_points)
        self.t12_buf = deque(maxlen=max_points)
        self.tamb_buf = deque(maxlen=max_points)

        self.port_var = tk.StringVar(value=args.port or "")
        self.baud_var = tk.StringVar(value=str(args.baud))
        self.connected_var = tk.StringVar(value="Disconnected")

        self.profile_var = tk.StringVar(value="staircase_characterization")
        self.sequence_var = tk.StringVar(value="5,0,-5,-10,-15,-18")
        self.tolerance_var = tk.StringVar(value="0.5")
        self.stable_dwell_var = tk.StringVar(value="300")
        self.post_dwell_var = tk.StringVar(value="300")
        self.max_hold_var = tk.StringVar(value="900")
        self.continue_timeout_var = tk.BooleanVar(value=True)
        self.slope_var = tk.StringVar(value="0.05")
        self.slope_window_var = tk.StringVar(value="60")
        self.manual_setpoint_var = tk.StringVar(value="0.0")

        self.recording_var = tk.StringVar(value="No")
        self.record_button_var = tk.StringVar(value="Start Recording")

        self.show_t3_var = tk.BooleanVar(value=True)
        self.show_t7_var = tk.BooleanVar(value=True)
        self.show_t12_var = tk.BooleanVar(value=True)
        self.show_tamb_var = tk.BooleanVar(value=False)

        self.status_vars = {
            "state": tk.StringVar(value="IDLE"),
            "target": tk.StringVar(value="--"),
            "plate": tk.StringVar(value="--"),
            "step": tk.StringVar(value="0 / 0"),
            "stabilized": tk.StringVar(value="No"),
            "elapsed": tk.StringVar(value="0 s"),
            "recording": self.recording_var,
            "fault": tk.StringVar(value="none"),
        }

        self.telemetry_vars = {
            "setpoint": tk.StringVar(value="--"),
            "power": tk.StringVar(value="--"),
            "pid": tk.StringVar(value="--"),
            "armed": tk.StringVar(value="--"),
            "T3": tk.StringVar(value="--"),
            "T7": tk.StringVar(value="--"),
            "T12": tk.StringVar(value="--"),
            "Tamb": tk.StringVar(value="--"),
            "filtered": tk.StringVar(value="--"),
            "slope": tk.StringVar(value="--"),
            "firmware": tk.StringVar(value="--"),
        }

        self._build_ui()
        self.refresh_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if args.port:
            self.root.after(300, self.connect)

        self.root.after(250, self.periodic_ui_update)

    def _initial_geometry(self) -> str:
        screen_w = max(1, self.root.winfo_screenwidth())
        screen_h = max(1, self.root.winfo_screenheight())
        width = min(1480, max(960, screen_w - 80))
        height = min(860, max(640, screen_h - 120))
        width = min(width, max(720, screen_w - 40))
        height = min(height, max(520, screen_h - 80))
        return f"{width}x{height}"

    def _update_right_panel_scrollregion(self, _event: Optional[tk.Event] = None) -> None:
        if self.right_panel_canvas is None:
            return
        bbox = self.right_panel_canvas.bbox("all")
        if bbox is not None:
            self.right_panel_canvas.configure(scrollregion=bbox)

    def _resize_right_panel_width(self, event: tk.Event) -> None:
        if self.right_panel_canvas is None or self.right_panel_window is None:
            return
        self.right_panel_canvas.itemconfigure(self.right_panel_window, width=event.width)

    def _widget_in_right_panel(self, widget: Any) -> bool:
        current = widget
        while current is not None:
            if current in (self.right_panel_canvas, self.right_panel_frame):
                return True
            current = getattr(current, "master", None)
        return False

    def _scroll_right_panel(self, units: int) -> None:
        if self.right_panel_canvas is None or units == 0:
            return
        self.right_panel_canvas.yview_scroll(units, "units")

    def _on_right_panel_mousewheel(self, event: tk.Event) -> None:
        if not self._widget_in_right_panel(getattr(event, "widget", None)):
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        units = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self._scroll_right_panel(units)

    def _on_right_panel_mousewheel_linux(self, event: tk.Event) -> None:
        if not self._widget_in_right_panel(getattr(event, "widget", None)):
            return
        num = getattr(event, "num", None)
        if num == 4:
            self._scroll_right_panel(-1)
        elif num == 5:
            self._scroll_right_panel(1)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        conn = ttk.LabelFrame(self.root, text="Connection")
        conn.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        for col in range(8):
            conn.columnconfigure(col, weight=1 if col in (1, 5) else 0)

        ttk.Label(conn, text="Port").grid(row=0, column=0, padx=5, pady=6, sticky="w")
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var, width=18, state="readonly")
        self.port_combo.grid(row=0, column=1, padx=5, pady=6, sticky="ew")

        ttk.Button(conn, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=5, pady=6)
        ttk.Label(conn, text="Baud").grid(row=0, column=3, padx=5, pady=6, sticky="w")
        ttk.Entry(conn, textvariable=self.baud_var, width=10).grid(row=0, column=4, padx=5, pady=6, sticky="w")
        ttk.Button(conn, text="Connect", command=self.connect).grid(row=0, column=5, padx=5, pady=6, sticky="ew")
        ttk.Button(conn, text="Disconnect", command=self.disconnect).grid(row=0, column=6, padx=5, pady=6, sticky="ew")
        ttk.Label(conn, textvariable=self.connected_var).grid(row=0, column=7, padx=5, pady=6, sticky="e")

        left = ttk.Frame(self.root)
        left.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        right_container = ttk.Frame(self.root)
        right_container.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))
        right_container.rowconfigure(0, weight=1)
        right_container.columnconfigure(0, weight=1)

        # Keep the controls column scrollable so smaller displays can reach the lower widgets.
        self.right_panel_canvas = tk.Canvas(right_container, highlightthickness=0, borderwidth=0)
        self.right_panel_canvas.grid(row=0, column=0, sticky="nsew")

        right_scroll = ttk.Scrollbar(right_container, orient="vertical", command=self.right_panel_canvas.yview)
        right_scroll.grid(row=0, column=1, sticky="ns")
        self.right_panel_canvas.configure(yscrollcommand=right_scroll.set)

        self.right_panel_frame = ttk.Frame(self.right_panel_canvas)
        self.right_panel_frame.columnconfigure(0, weight=1)
        self.right_panel_window = self.right_panel_canvas.create_window((0, 0), window=self.right_panel_frame, anchor="nw")
        self.right_panel_frame.bind("<Configure>", self._update_right_panel_scrollregion)
        self.right_panel_canvas.bind("<Configure>", self._resize_right_panel_width)

        self.root.bind_all("<MouseWheel>", self._on_right_panel_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_right_panel_mousewheel_linux, add="+")
        self.root.bind_all("<Button-5>", self._on_right_panel_mousewheel_linux, add="+")

        self._build_plot(left)
        self._build_status_panel(self.right_panel_frame)
        self._build_telemetry_panel(self.right_panel_frame)
        self._build_manual_controls(self.right_panel_frame)
        self._build_characterization_controls(self.right_panel_frame)
        self._build_log_panel(self.right_panel_frame)
        self.root.after_idle(self._update_right_panel_scrollregion)

    def _build_plot(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Live Plot")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8.5, 6.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Cryostage temperatures")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Temperature (C)")
        self.ax.grid(True, alpha=0.3)

        (self.plate_line,) = self.ax.plot([], [], color="black", linewidth=2.0, label="Plate")
        (self.set_line,) = self.ax.plot([], [], color="tab:red", linestyle="--", linewidth=1.8, label="Setpoint")
        (self.t3_line,) = self.ax.plot([], [], color="tab:blue", linewidth=1.2, label="T3")
        (self.t7_line,) = self.ax.plot([], [], color="tab:orange", linewidth=1.2, label="T7")
        (self.t12_line,) = self.ax.plot([], [], color="tab:green", linewidth=1.2, label="T12")
        (self.tamb_line,) = self.ax.plot([], [], color="tab:purple", linewidth=1.2, label="Tamb")
        self.ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.draw_idle()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        toggles = ttk.Frame(frame)
        toggles.grid(row=1, column=0, sticky="w", padx=6, pady=(0, 6))
        ttk.Checkbutton(toggles, text="T3", variable=self.show_t3_var, command=self.update_plot).grid(row=0, column=0, padx=4)
        ttk.Checkbutton(toggles, text="T7", variable=self.show_t7_var, command=self.update_plot).grid(row=0, column=1, padx=4)
        ttk.Checkbutton(toggles, text="T12", variable=self.show_t12_var, command=self.update_plot).grid(row=0, column=2, padx=4)
        ttk.Checkbutton(toggles, text="Tamb", variable=self.show_tamb_var, command=self.update_plot).grid(row=0, column=3, padx=4)

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Status")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        rows = [
            ("State", self.status_vars["state"]),
            ("Target (C)", self.status_vars["target"]),
            ("Plate (C)", self.status_vars["plate"]),
            ("Step", self.status_vars["step"]),
            ("Stabilized", self.status_vars["stabilized"]),
            ("Elapsed", self.status_vars["elapsed"]),
            ("Recording", self.status_vars["recording"]),
            ("Fault", self.status_vars["fault"]),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
            ttk.Label(frame, textvariable=var).grid(row=row, column=1, sticky="w", padx=6, pady=2)

    def _build_telemetry_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Live Telemetry")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        rows = [
            ("Setpoint", self.telemetry_vars["setpoint"]),
            ("Power (%)", self.telemetry_vars["power"]),
            ("PID", self.telemetry_vars["pid"]),
            ("Armed", self.telemetry_vars["armed"]),
            ("T3", self.telemetry_vars["T3"]),
            ("T7", self.telemetry_vars["T7"]),
            ("T12", self.telemetry_vars["T12"]),
            ("Tamb", self.telemetry_vars["Tamb"]),
            ("Filtered", self.telemetry_vars["filtered"]),
            ("Slope", self.telemetry_vars["slope"]),
            ("Firmware", self.telemetry_vars["firmware"]),
        ]
        for row, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
            ttk.Label(frame, textvariable=var).grid(row=row, column=1, sticky="w", padx=6, pady=2)

    def _build_manual_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Manual Controls")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for col in range(2):
            frame.columnconfigure(col, weight=1)

        ttk.Button(frame, text="Arm", command=lambda: self.send_and_query("arm 1")).grid(row=0, column=0, padx=6, pady=4, sticky="ew")
        ttk.Button(frame, text="Disarm", command=lambda: self.send_and_query("arm 0")).grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        ttk.Button(frame, text="PID On", command=lambda: self.send_and_query("pid on")).grid(row=1, column=0, padx=6, pady=4, sticky="ew")
        ttk.Button(frame, text="PID Off", command=lambda: self.send_and_query("pid off")).grid(row=1, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(frame, text="Manual setpoint (C)").grid(row=2, column=0, padx=6, pady=(8, 2), sticky="w")
        ttk.Entry(frame, textvariable=self.manual_setpoint_var).grid(row=2, column=1, padx=6, pady=(8, 2), sticky="ew")
        ttk.Button(frame, text="Apply Setpoint", command=self.apply_manual_setpoint).grid(row=3, column=0, columnspan=2, padx=6, pady=4, sticky="ew")
        ttk.Button(frame, textvariable=self.record_button_var, command=self.toggle_recording).grid(
            row=4, column=0, columnspan=2, padx=6, pady=(8, 4), sticky="ew"
        )

    def _build_characterization_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Characterization")
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Profile name").grid(row=0, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.profile_var).grid(row=0, column=1, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text="Sequence (C)").grid(row=1, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.sequence_var).grid(row=1, column=1, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text="Tolerance").grid(row=2, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.tolerance_var).grid(row=2, column=1, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text="Stable dwell (s)").grid(row=3, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.stable_dwell_var).grid(row=3, column=1, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text="Post dwell (s)").grid(row=4, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.post_dwell_var).grid(row=4, column=1, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text="Max hold (s)").grid(row=5, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.max_hold_var).grid(row=5, column=1, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text="Slope limit").grid(row=6, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.slope_var).grid(row=6, column=1, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text="Slope window (s)").grid(row=7, column=0, padx=6, pady=2, sticky="w")
        ttk.Entry(frame, textvariable=self.slope_window_var).grid(row=7, column=1, padx=6, pady=2, sticky="ew")
        ttk.Checkbutton(frame, text="Continue on timeout", variable=self.continue_timeout_var).grid(
            row=8, column=0, columnspan=2, padx=6, pady=4, sticky="w"
        )
        ttk.Button(frame, text="Apply Settings", command=self.apply_characterization_settings).grid(
            row=9, column=0, columnspan=2, padx=6, pady=(8, 4), sticky="ew"
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=10, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 4))
        for col in range(2):
            buttons.columnconfigure(col, weight=1)
        ttk.Button(buttons, text="Start Characterization", command=self.start_characterization).grid(
            row=0, column=0, columnspan=2, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(buttons, text="Pause", command=lambda: self.send_and_query("char pause")).grid(
            row=1, column=0, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(buttons, text="Resume", command=lambda: self.send_and_query("char resume")).grid(
            row=1, column=1, padx=4, pady=4, sticky="ew"
        )
        ttk.Button(buttons, text="Stop / Abort", command=self.stop_characterization).grid(
            row=2, column=0, columnspan=2, padx=4, pady=4, sticky="ew"
        )

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Panel Log")
        frame.grid(row=5, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(frame, height=10, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)

        log_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 6), pady=6)
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def append_log(self, message: str) -> None:
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_messages.put(f"[{timestamp}] {message}")

    def flush_log_messages(self) -> None:
        pending = []
        while True:
            try:
                pending.append(self.log_messages.get_nowait())
            except queue.Empty:
                break
        if not pending:
            return

        self.log_text.configure(state="normal")
        for message in pending:
            self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def refresh_ports(self) -> None:
        ports = list_serial_port_names()
        self.port_combo["values"] = ports
        if not self.port_var.get() and ports:
            self.port_var.set(ports[0])

    def is_connected(self) -> bool:
        return self.serial_port is not None and self.serial_port.is_open

    def connect(self) -> None:
        if self.is_connected():
            return

        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Connection", "Select a serial port first.")
            return

        try:
            baud = int(self.baud_var.get().strip())
        except ValueError:
            messagebox.showerror("Connection", "Baud rate must be an integer.")
            return

        try:
            ser = serial.Serial(port, baud, timeout=0.2)
            ser.reset_input_buffer()
        except Exception as exc:
            messagebox.showerror("Connection", f"Could not open {port}: {exc}")
            return

        self.serial_port = ser
        self.stop_reader.clear()
        self.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
        self.reader_thread.start()
        self.connected_var.set(f"Connected: {port}")
        self.append_log(f"Connected to {port} @ {baud}")
        self.send_command("stream 1")
        self.send_command("status")
        self.send_command("char config")

    def disconnect(self) -> None:
        self.stop_reader.set()
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.serial_port = None
        self.connected_var.set("Disconnected")

    def send_command(self, command: str) -> None:
        if not self.is_connected():
            return
        with self.tx_lock:
            assert self.serial_port is not None
            self.serial_port.write((command.strip() + "\r\n").encode("utf-8", errors="ignore"))
            self.serial_port.flush()

    def send_and_query(self, command: str) -> None:
        if not self.is_connected():
            messagebox.showwarning("Command", "Connect to the firmware first.")
            return
        self.send_command(command)
        self.send_command("status")

    def apply_manual_setpoint(self) -> None:
        try:
            value = float(self.manual_setpoint_var.get().strip())
        except ValueError:
            messagebox.showerror("Setpoint", "Manual setpoint must be numeric.")
            return
        self.send_and_query(f"set {value}")

    def apply_characterization_settings(self) -> bool:
        if not self.is_connected():
            messagebox.showwarning("Characterization", "Connect to the firmware first.")
            return False

        commands = [
            f"char seq {self.sequence_var.get().strip()}",
            f"char tol {self.tolerance_var.get().strip()}",
            f"char dwell {self.stable_dwell_var.get().strip()}",
            f"char postdwell {self.post_dwell_var.get().strip()}",
            f"char maxhold {self.max_hold_var.get().strip()}",
            f"char continue_timeout {1 if self.continue_timeout_var.get() else 0}",
            f"char slope {self.slope_var.get().strip()}",
            f"char slope_window {self.slope_window_var.get().strip()}",
            "char config",
        ]
        for command in commands:
            self.send_command(command)
            time.sleep(0.04)
        self.append_log("Characterization settings pushed to firmware")
        return True

    def start_characterization(self) -> None:
        if not self.apply_characterization_settings():
            return
        self.send_command("char start")
        self.send_command("char status")

    def stop_characterization(self) -> None:
        self.send_command("char stop")
        self.send_command("char status")

    def build_metadata(self) -> Dict[str, Any]:
        latest = dict(self.latest)
        return {
            "date_time": dt.datetime.now().isoformat(timespec="seconds"),
            "com_port": self.port_var.get().strip(),
            "baud_rate": self.baud_var.get().strip(),
            "profile_name": self.profile_var.get().strip(),
            "setpoint_sequence": self.sequence_var.get().strip(),
            "tolerance_band_c": self.tolerance_var.get().strip(),
            "stabilization_dwell_s": self.stable_dwell_var.get().strip(),
            "post_stabilization_dwell_s": self.post_dwell_var.get().strip(),
            "max_hold_s": self.max_hold_var.get().strip(),
            "continue_on_timeout": int(self.continue_timeout_var.get()),
            "slope_limit_c_per_min": self.slope_var.get().strip(),
            "slope_window_s": self.slope_window_var.get().strip(),
            "pid_kp": latest.get("kp", ""),
            "pid_ki": latest.get("ki", ""),
            "pid_kd": latest.get("kd", ""),
            "firmware_version": latest.get("fw", ""),
        }

    def toggle_recording(self) -> None:
        if self.logger.recording:
            self.logger.stop()
            self.record_button_var.set("Start Recording")
            self.recording_var.set("No")
            self.append_log("CSV recording stopped")
            return

        default_name = f"cryostage_characterization_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
        chosen = filedialog.asksaveasfilename(
            title="Save characterization CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not chosen:
            return

        path = Path(chosen)
        self.logger.start(path, self.build_metadata())
        self.record_button_var.set("Stop Recording")
        self.recording_var.set("Yes")
        self.append_log(f"CSV recording started: {path}")

        with self.lock:
            snapshot = dict(self.latest)
        if snapshot:
            self.logger.write_event("recording_started", "", snapshot)

    def reader_loop(self) -> None:
        assert self.serial_port is not None
        ser = self.serial_port
        while not self.stop_reader.is_set():
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                telemetry = parse_telemetry_line(line)
                if telemetry is not None:
                    self.handle_telemetry(telemetry)
                    continue

                status = parse_pipe_kv_line(line, "[STATUS]")
                if status is not None:
                    with self.lock:
                        self.latest.update(status)
                    continue

                char_status = parse_pipe_kv_line(line, "[CHAR]")
                if char_status is not None:
                    with self.lock:
                        self.latest.update(char_status)
                    continue

                char_cfg = parse_pipe_kv_line(line, "[CHARCFG]")
                if char_cfg is not None:
                    with self.lock:
                        self.latest.update(char_cfg)
                    continue

                char_seq = parse_charseq_line(line)
                if char_seq is not None:
                    with self.lock:
                        self.latest["sequence_reported"] = char_seq
                    continue

                self.append_log(line)
            except Exception as exc:
                if not self.stop_reader.is_set():
                    self.pending_error = str(exc)
                break

    def handle_telemetry(self, data: Dict[str, Any]) -> None:
        ts_ms = as_float(data.get("ts"))
        if not math.isfinite(ts_ms):
            return

        with self.lock:
            if self.last_ts_ms is not None and ts_ms + 1000.0 < self.last_ts_ms:
                self.panel_t0_ms = ts_ms
                self.t_buf.clear()
                self.plate_buf.clear()
                self.set_buf.clear()
                self.t3_buf.clear()
                self.t7_buf.clear()
                self.t12_buf.clear()
                self.tamb_buf.clear()

            if self.panel_t0_ms is None:
                self.panel_t0_ms = ts_ms

            self.last_ts_ms = ts_ms
            panel_t_s = (ts_ms - self.panel_t0_ms) / 1000.0
            snapshot = {"panel_t_s": panel_t_s, **data}

            self.latest = dict(self.latest)
            self.latest.update(snapshot)

            self.t_buf.append(panel_t_s)
            self.plate_buf.append(as_float(snapshot.get("T_cal")))
            self.set_buf.append(as_float(snapshot.get("set")))
            self.t3_buf.append(as_float(snapshot.get("T3")))
            self.t7_buf.append(as_float(snapshot.get("T7")))
            self.t12_buf.append(as_float(snapshot.get("T12")))
            self.tamb_buf.append(as_float(snapshot.get("Tamb")))

            current_snapshot = dict(self.latest)
            previous_snapshot = dict(self.last_snapshot)
            self.last_snapshot = current_snapshot

        if self.logger.recording:
            self.logger.write_telemetry(current_snapshot)
            self.detect_events(previous_snapshot, current_snapshot)

    def detect_events(self, previous: Dict[str, Any], current: Dict[str, Any]) -> None:
        prev_state = str(previous.get("char_state", ""))
        curr_state = str(current.get("char_state", ""))
        prev_step = as_int(previous.get("char_step"))
        curr_step = as_int(current.get("char_step"))

        if prev_state != curr_state:
            if curr_state == "STARTING":
                self.logger.write_event("characterization_started", "", current)
            elif curr_state == "PAUSED":
                self.logger.write_event("characterization_paused", "", current)
            elif prev_state == "PAUSED" and curr_state != "PAUSED":
                self.logger.write_event("characterization_resumed", "", current)
            elif curr_state == "FINISHED":
                self.logger.write_event("characterization_finished", "", current)
            elif curr_state == "ABORTED":
                self.logger.write_event("characterization_aborted", "", current)
            elif curr_state == "FAULT":
                self.logger.write_event("fault_occurred", str(current.get("fault_msg", "")), current)

        if curr_step != prev_step or current.get("char_target") != previous.get("char_target"):
            if curr_step > 0:
                detail = f"step={curr_step}/{as_int(current.get('char_total'))}, target={current.get('char_target')}"
                self.logger.write_event("step_changed", detail, current)

        if as_int(previous.get("stable")) == 0 and as_int(current.get("stable")) == 1:
            self.logger.write_event("stabilized_achieved", "", current)

        if as_int(previous.get("timeout")) == 0 and as_int(current.get("timeout")) == 1:
            self.logger.write_event("timeout", "", current)

        if as_int(previous.get("fault")) == 0 and as_int(current.get("fault")) == 1:
            self.logger.write_event("fault_occurred", str(current.get("fault_msg", "")), current)

    def periodic_ui_update(self) -> None:
        self.flush_log_messages()

        if self.pending_error:
            message = self.pending_error
            self.pending_error = None
            self.disconnect()
            messagebox.showerror("Serial connection", f"Reader stopped: {message}")

        with self.lock:
            latest = dict(self.latest)

        self.status_vars["state"].set(str(latest.get("char_state", "IDLE")))
        self.status_vars["target"].set(format_number(latest.get("char_target"), 2))
        self.status_vars["plate"].set(format_number(latest.get("T_cal"), 3))
        self.status_vars["step"].set(f"{as_int(latest.get('char_step'))} / {as_int(latest.get('char_total'))}")
        self.status_vars["stabilized"].set("Yes" if as_int(latest.get("stable")) else "No")
        self.status_vars["elapsed"].set(f"{format_number(latest.get('char_time_s'), 1)} s")
        fault_text = str(latest.get("fault_msg", "none")) if as_int(latest.get("fault")) else "none"
        self.status_vars["fault"].set(fault_text)

        self.telemetry_vars["setpoint"].set(format_number(latest.get("set"), 2))
        self.telemetry_vars["power"].set(format_number(latest.get("power"), 1))
        self.telemetry_vars["pid"].set("On" if as_int(latest.get("pid")) else "Off")
        self.telemetry_vars["armed"].set("Yes" if as_int(latest.get("armed")) else "No")
        self.telemetry_vars["T3"].set(format_number(latest.get("T3"), 3))
        self.telemetry_vars["T7"].set(format_number(latest.get("T7"), 3))
        self.telemetry_vars["T12"].set(format_number(latest.get("T12"), 3))
        self.telemetry_vars["Tamb"].set(format_number(latest.get("Tamb"), 3))
        self.telemetry_vars["filtered"].set(format_number(latest.get("char_filtered"), 3))
        self.telemetry_vars["slope"].set(format_number(latest.get("char_slope"), 4))
        self.telemetry_vars["firmware"].set(str(latest.get("fw", "--")))

        self.update_plot()
        self.root.after(250, self.periodic_ui_update)

    def update_plot(self) -> None:
        with self.lock:
            t = list(self.t_buf)
            plate = list(self.plate_buf)
            setpoint = list(self.set_buf)
            t3 = list(self.t3_buf)
            t7 = list(self.t7_buf)
            t12 = list(self.t12_buf)
            tamb = list(self.tamb_buf)

        self.plate_line.set_data(t, plate)
        self.set_line.set_data(t, setpoint)
        self.t3_line.set_data(t, t3)
        self.t7_line.set_data(t, t7)
        self.t12_line.set_data(t, t12)
        self.tamb_line.set_data(t, tamb)

        self.t3_line.set_visible(self.show_t3_var.get())
        self.t7_line.set_visible(self.show_t7_var.get())
        self.t12_line.set_visible(self.show_t12_var.get())
        self.tamb_line.set_visible(self.show_tamb_var.get())

        if t:
            x_max = t[-1] + 1.0
            x_min = max(0.0, x_max - self.args.buffer_sec)
            self.ax.set_xlim(x_min, x_max)

            y_values = []
            for series in (plate, setpoint):
                y_values.extend(v for v in series if math.isfinite(v))
            if self.show_t3_var.get():
                y_values.extend(v for v in t3 if math.isfinite(v))
            if self.show_t7_var.get():
                y_values.extend(v for v in t7 if math.isfinite(v))
            if self.show_t12_var.get():
                y_values.extend(v for v in t12 if math.isfinite(v))
            if self.show_tamb_var.get():
                y_values.extend(v for v in tamb if math.isfinite(v))

            if y_values:
                y_min = min(y_values)
                y_max = max(y_values)
                margin = max(1.0, 0.08 * max(1.0, y_max - y_min))
                self.ax.set_ylim(y_min - margin, y_max + margin)

        self.canvas.draw_idle()

    def on_close(self) -> None:
        try:
            if self.logger.recording:
                with self.lock:
                    snapshot = dict(self.latest)
                self.logger.write_event("panel_closed", "", snapshot)
        finally:
            self.logger.stop()
            self.disconnect()
            self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cryostage characterization desktop panel")
    parser.add_argument("--port", default=None, help="Serial port to preselect/connect")
    parser.add_argument("--baud", default=115200, type=int, help="Serial baud rate")
    parser.add_argument("--buffer-sec", default=1800.0, type=float, help="Visible plot history in seconds")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = tk.Tk()
    panel = CryostagePanel(root, args)
    panel.append_log("Panel ready")
    root.mainloop()


if __name__ == "__main__":
    main()

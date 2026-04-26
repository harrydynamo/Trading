"""
Trading Suite Launcher
----------------------
Tkinter GUI that lets you pick and launch any of the three trading apps.
Each app runs as a Streamlit server on a dedicated port; the browser opens automatically.

Apps:
  - Live Trading UI    → trading_ui/app.py   (port 8501)
  - Stock Screener UI  → ui/app.py           (port 8502)
  - Live Signals       → live_signals/run.py (terminal, no browser)
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
import urllib.request

# ── Resolve project root ──────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    ROOT   = sys._MEIPASS
    PYTHON = sys.executable
else:
    ROOT   = os.path.dirname(os.path.abspath(__file__))
    PYTHON = sys.executable

os.chdir(ROOT)

# ── App definitions ───────────────────────────────────────────────────────────
APPS = {
    "📊  India Trading UI": {
        "script": os.path.join(ROOT, "trading_ui", "app.py"),
        "port":   8501,
        "desc":   "NSE/BSE live chart, signals, trade setups and options chain",
        "mode":   "streamlit",
    },
    "🇺🇸  US Trading UI": {
        "script": os.path.join(ROOT, "trading_ui_us", "app.py"),
        "port":   8503,
        "desc":   "NYSE/NASDAQ live chart, signals, trade setups and options chain",
        "mode":   "streamlit",
    },
    "🔎  Stock Screener": {
        "script": os.path.join(ROOT, "ui", "app.py"),
        "port":   8502,
        "desc":   "Score and rank NSE/BSE stocks by fundamentals",
        "mode":   "streamlit",
    },
    "⚡  Live Signals (Terminal)": {
        "script": os.path.join(ROOT, "live_signals", "run.py"),
        "port":   None,
        "desc":   "Scan top-50 watchlist for real-time BUY/EXIT signals",
        "mode":   "terminal",
    },
}

_procs: dict[str, subprocess.Popen] = {}
_log_var: tk.StringVar | None = None   # status label StringVar


def _log(msg: str):
    """Update the status bar text (thread-safe)."""
    if _log_var:
        _log_var.set(msg)


def _streamlit_cmd(script: str, port: int) -> list[str]:
    return [
        PYTHON, "-m", "streamlit", "run", script,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ]


def _wait_and_open(port: int, btn: tk.Button, name: str):
    """Background thread: poll until server is up, then open browser."""
    url      = f"http://localhost:{port}"
    deadline = time.time() + 90   # wait up to 90 seconds

    _log(f"Starting server on port {port}…")

    # Give process a moment to spin up before polling
    time.sleep(4)

    while time.time() < deadline:
        # Check if process died already
        proc = _procs.get(name)
        if proc and proc.poll() is not None:
            _log(f"❌  Server failed to start. Check that streamlit is installed: pip install streamlit")
            return

        try:
            urllib.request.urlopen(url, timeout=2)
            break   # server responded
        except Exception:
            _log(f"Waiting for server… ({int(deadline - time.time())}s left)")
            time.sleep(2)
    else:
        _log(f"⚠️  Timed out. Try opening manually: {url}")
        return

    _log(f"✅  Running at {url}  — opening browser…")
    try:
        # os.startfile is the most reliable way to open a URL on Windows
        if sys.platform == "win32":
            os.startfile(url)
        else:
            webbrowser.open(url)
    except Exception:
        _log(f"✅  Open your browser and go to: {url}")


def launch(name: str, btn: tk.Button):
    app = APPS[name]

    # Already running — just open browser
    if name in _procs and _procs[name].poll() is None:
        if app["mode"] == "streamlit":
            url = f"http://localhost:{app['port']}"
            try:
                if sys.platform == "win32":
                    os.startfile(url)
                else:
                    webbrowser.open(url)
            except Exception:
                _log(f"Open your browser and go to: {url}")
        return

    if app["mode"] == "streamlit":
        cmd = _streamlit_cmd(app["script"], app["port"])

        # Write stdout/stderr to a log file so errors are visible
        log_path = os.path.join(ROOT, f"streamlit_{app['port']}.log")
        log_file = open(log_path, "w")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except FileNotFoundError:
            _log("❌  Python not found. Make sure Python is in PATH.")
            return

        _procs[name] = proc
        btn.config(text="🟢  Open in Browser", bg="#1a7a3c", fg="white")

        threading.Thread(
            target=_wait_and_open,
            args=(app["port"], btn, name),
            daemon=True,
        ).start()

    elif app["mode"] == "terminal":
        if sys.platform == "win32":
            subprocess.Popen(
                f'start cmd /k "{PYTHON}" "{app["script"]}"',
                shell=True, cwd=ROOT,
            )
        elif sys.platform == "darwin":
            osa = f'tell application "Terminal" to do script "cd {ROOT} && {PYTHON} {app["script"]}"'
            subprocess.Popen(["osascript", "-e", osa])
        else:
            subprocess.Popen(
                ["x-terminal-emulator", "-e", f"{PYTHON} {app['script']}"],
                cwd=ROOT,
            )
        _log("Launched in new terminal window")


def stop_all():
    for name, proc in list(_procs.items()):
        if proc.poll() is None:
            proc.terminate()
    _procs.clear()
    messagebox.showinfo("Stopped", "All running apps have been stopped.")


def build_ui(root: tk.Tk):
    global _log_var

    root.title("Trading Suite")
    root.resizable(False, False)
    root.configure(bg="#f8f9fa")

    try:
        ico = os.path.join(ROOT, "trading_ui", "icon.ico")
        if os.path.exists(ico):
            root.iconbitmap(ico)
    except Exception:
        pass

    # ── Header ───────────────────────────────────────────────────────────────
    hdr = tk.Frame(root, bg="#1a3a6e", pady=16)
    hdr.pack(fill="x")
    tk.Label(hdr, text="📈  Trading Suite", font=("Segoe UI", 20, "bold"),
             bg="#1a3a6e", fg="white").pack()
    tk.Label(hdr, text="NSE / BSE — Live Charts · Screener · Signals",
             font=("Segoe UI", 10), bg="#1a3a6e", fg="#aac4f0").pack()

    # ── App cards ────────────────────────────────────────────────────────────
    card_frame = tk.Frame(root, bg="#f8f9fa", padx=24, pady=20)
    card_frame.pack(fill="both", expand=True)

    for name, app in APPS.items():
        card = tk.Frame(card_frame, bg="white", relief="flat",
                        highlightbackground="#dee2e6", highlightthickness=1,
                        pady=12, padx=16)
        card.pack(fill="x", pady=6)

        left = tk.Frame(card, bg="white")
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=name, font=("Segoe UI", 12, "bold"),
                 bg="white", fg="#111").pack(anchor="w")
        tk.Label(left, text=app["desc"], font=("Segoe UI", 9),
                 bg="white", fg="#666").pack(anchor="w")
        if app["port"]:
            tk.Label(left, text=f"localhost:{app['port']}",
                     font=("Segoe UI", 8), bg="white", fg="#aaa").pack(anchor="w")

        btn = tk.Button(
            card,
            text="▶  Launch",
            font=("Segoe UI", 10, "bold"),
            bg="#1a5aad", fg="white",
            relief="flat", padx=16, pady=6, cursor="hand2",
            activebackground="#1446a0", activeforeground="white",
        )
        btn.config(command=lambda n=name, b=btn: launch(n, b))
        btn.pack(side="right", padx=4)

    # ── Status bar ───────────────────────────────────────────────────────────
    ttk.Separator(root, orient="horizontal").pack(fill="x")

    bottom = tk.Frame(root, bg="#f0f2f5", pady=8, padx=16)
    bottom.pack(fill="x")

    _log_var = tk.StringVar(value="Select an app to launch")
    tk.Label(bottom, textvariable=_log_var, font=("Segoe UI", 9),
             bg="#f0f2f5", fg="#555", wraplength=380, justify="left").pack(side="left")
    tk.Button(bottom, text="⏹ Stop All", font=("Segoe UI", 9),
              bg="#c0392b", fg="white", relief="flat", padx=10, pady=3,
              cursor="hand2", command=stop_all).pack(side="right")


def main():
    root = tk.Tk()
    root.geometry("520x500")
    build_ui(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_all(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()

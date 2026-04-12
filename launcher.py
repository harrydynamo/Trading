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

# ── Resolve project root (works both as .py and as frozen .exe) ───────────────
if getattr(sys, "frozen", False):
    ROOT = sys._MEIPASS          # PyInstaller temp folder
    PYTHON = sys.executable      # the bundled python
else:
    ROOT   = os.path.dirname(os.path.abspath(__file__))
    PYTHON = sys.executable

os.chdir(ROOT)   # ensure relative imports work


# ── App definitions ───────────────────────────────────────────────────────────
APPS = {
    "📊  Live Trading UI": {
        "script": os.path.join(ROOT, "trading_ui", "app.py"),
        "port":   8501,
        "desc":   "Live chart, signals, backtest and options chain",
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

# Track running processes
_procs: dict[str, subprocess.Popen] = {}


def _streamlit_cmd(script: str, port: int) -> list[str]:
    return [
        PYTHON, "-m", "streamlit", "run", script,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ]


def _open_browser(port: int, delay: float = 3.0):
    """Wait for server to start, then open browser."""
    def _wait_and_open():
        time.sleep(delay)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=_wait_and_open, daemon=True).start()


def launch(name: str, btn: tk.Button, status_var: tk.StringVar):
    app = APPS[name]

    # If already running, just open browser
    if name in _procs and _procs[name].poll() is None:
        if app["mode"] == "streamlit":
            webbrowser.open(f"http://localhost:{app['port']}")
        return

    if app["mode"] == "streamlit":
        cmd  = _streamlit_cmd(app["script"], app["port"])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
        )
        _procs[name] = proc
        _open_browser(app["port"])
        btn.config(text="🟢  Open in Browser", bg="#1a7a3c", fg="white")
        status_var.set(f"Started on http://localhost:{app['port']}")

    elif app["mode"] == "terminal":
        # Open a new terminal window
        if sys.platform == "win32":
            subprocess.Popen(
                ["cmd", "/c", "start", "cmd", "/k",
                 PYTHON, app["script"]],
                cwd=ROOT,
            )
        elif sys.platform == "darwin":
            app_script = app["script"]
            osa = f'tell application "Terminal" to do script "cd {ROOT} && {PYTHON} {app_script}"'
            subprocess.Popen(["osascript", "-e", osa])
        else:
            subprocess.Popen(
                ["x-terminal-emulator", "-e",
                 f"{PYTHON} {app['script']}"],
                cwd=ROOT,
            )
        status_var.set("Launched in new terminal window")


def stop_all():
    for name, proc in list(_procs.items()):
        if proc.poll() is None:
            proc.terminate()
    _procs.clear()
    messagebox.showinfo("Stopped", "All running apps have been stopped.")


def build_ui(root: tk.Tk):
    root.title("Trading Suite")
    root.resizable(False, False)
    root.configure(bg="#f8f9fa")

    # Try to set window icon (ignored gracefully if missing)
    try:
        ico = os.path.join(ROOT, "trading_ui", "icon.ico")
        if os.path.exists(ico):
            root.iconbitmap(ico)
    except Exception:
        pass

    # ── Header ──────────────────────────────────────────────────────────────
    hdr = tk.Frame(root, bg="#1a3a6e", pady=16)
    hdr.pack(fill="x")
    tk.Label(hdr, text="📈  Trading Suite", font=("Segoe UI", 20, "bold"),
             bg="#1a3a6e", fg="white").pack()
    tk.Label(hdr, text="NSE / BSE — Live Charts · Screener · Signals",
             font=("Segoe UI", 10), bg="#1a3a6e", fg="#aac4f0").pack()

    # ── App cards ───────────────────────────────────────────────────────────
    card_frame = tk.Frame(root, bg="#f8f9fa", padx=24, pady=20)
    card_frame.pack(fill="both", expand=True)

    status_var = tk.StringVar(value="Select an app to launch")

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
        btn.config(command=lambda n=name, b=btn, s=status_var: launch(n, b, s))
        btn.pack(side="right", padx=4)

    # ── Status bar ──────────────────────────────────────────────────────────
    sep = ttk.Separator(root, orient="horizontal")
    sep.pack(fill="x", padx=0)

    bottom = tk.Frame(root, bg="#f0f2f5", pady=8, padx=16)
    bottom.pack(fill="x")
    tk.Label(bottom, textvariable=status_var, font=("Segoe UI", 9),
             bg="#f0f2f5", fg="#555").pack(side="left")
    tk.Button(bottom, text="⏹ Stop All", font=("Segoe UI", 9),
              bg="#c0392b", fg="white", relief="flat", padx=10, pady=3,
              cursor="hand2", command=stop_all).pack(side="right")


def main():
    root = tk.Tk()
    root.geometry("520x400")
    build_ui(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_all(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()

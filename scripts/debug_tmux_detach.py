#!/usr/bin/env python3
import os
import signal
import sys
import time
from datetime import datetime
from tqdm import tqdm

LOG_FILE = "signals.log"

def log_event(message: str):
    """Log directly to stderr (unbuffered write) and persist to disk."""
    timestamp = datetime.now().isoformat()
    entry = f"\n[{timestamp}] [PID {os.getpid()}] {message}\n"

    # Write directly to stderr descriptor to avoid Python stdout buffering
    sys.stderr.write(entry)
    sys.stderr.flush()

    with open(LOG_FILE, "a") as f:
        f.write(entry)
        f.flush()

def signal_handler(signum, frame):
    try:
        sig_name = signal.Signals(signum).name
    except ValueError:
        sig_name = f"UNKNOWN({signum})"

    log_event(f"CAUGHT SIGNAL: {sig_name} (Code {signum})")

    if signum in (signal.SIGINT, signal.SIGTERM):
        log_event(f"Exiting on {sig_name}...")
        sys.exit(128 + signum)

def setup_signals():
    uncatchable = {signal.SIGKILL, signal.SIGSTOP}
    for sig in signal.valid_signals():
        if sig in uncatchable:
            continue
        try:
            signal.signal(sig, signal_handler)
        except (OSError, RuntimeError):
            pass

def main():
    setup_signals()
    log_event("Starting buffer stress test + signal listener...")

    # High iterations + tiny payloads to flood terminal line buffers fast
    total_steps = 100_000_000

    with tqdm(
        total=total_steps,
        desc="Stress-testing stdout buffer",
        unit="samples",
        mininterval=0.001,  # Force rapid UI redraws to flood the pty
        dynamic_ncols=True,
        file=sys.stderr     # Route tqdm directly through stderr
    ) as pbar:
        for i in range(total_steps):
            # Periodically append postfix metadata to bloat the byte volume per line
            if i % 100 == 0:
                pbar.set_postfix_str(f"bytes_spammed={i * 256}B | status=running")
            pbar.update(1)

    log_event("Test completed successfully.")

if __name__ == "__main__":
    main()

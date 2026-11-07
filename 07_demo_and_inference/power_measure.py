"""
power_measure.py

WHAT THIS DOES
    Measures what running the model costs in power on a Jetson, by
    comparing the board while it does nothing with the board while it runs
    the model.

        1. IDLE  (first)   the board sits still        --idle-seconds
        2. RUNNING         the inference command runs  --repeat times
        3. IDLE  (second)  the board sits still again

    Power is read once every half second from tegrastats, the tool NVIDIA
    ships with the board. The model's cost is the difference:

        inference power  =  running power  -  idle power

WHY IDLE TWICE
    A reading taken before the run can still include the previous program
    shutting down, and a board that has warmed up draws a little more than a
    cold one. Measuring idle twice shows which one is clean: the script
    uses the idle reading that varied LESS (the steadier one) and says so.

WHICH NUMBER IS "THE BOARD"
    tegrastats prints several rails. VDD_IN (POM_5V_IN on older boards) is
    the whole board, and already contains the CPU and GPU rails, so it is
    used alone. Summing all rails would count the same power twice. The
    individual rails are printed too, to show where the change happened.

THE FORMULAS (all printed)
    inference power      P_inf   = P_running - P_idle                  [mW]
    time per window      t_win   = run seconds / windows scored        [s]
    energy per window    E_win   = P_inf x t_win                       [mJ]
    charge per window    Q_win   = E_win / 1000 / V / 3.6              [mAh]
    on a worn device, one window every 30 s:
    busy fraction        duty    = t_win / 30
    added power          P_added = P_inf x duty                        [mW]
    added charge per day         = P_added / V x 24                    [mAh]
    (1 mAh at 1 V is 3.6 J, which is where the 3.6 comes from. The mAh
    figure depends on the supply voltage V, so V is printed beside it. mJ
    is what is actually measured.)

HOW TO RUN (on the Jetson, in this folder)
    sudo nvpmodel -q                        # note the power mode
    python3 power_measure.py \\
        --cmd "python3 55_session_checkpoints.py --pack demo_sessions_pack \\
               --no-explain --out run_output" \\
        --windows 569 --repeat 5 --idle-seconds 60

    --windows  is the number of windows ONE run of the command scores (569)
    --repeat   runs the command this many times back to back, so the board
               is busy long enough to be measured properly
    If tegrastats needs root on your board, run the whole thing with sudo.

OUTPUT
    power_summary.txt, power_trace.csv  (every half-second reading)
"""
import argparse
import csv
import os
import re
import statistics as st
import subprocess
import sys
import threading
import time

RAIL = re.compile(r"(\w+)\s+(\d+)mW/(\d+)mW")
WHOLE_BOARD = ("VDD_IN", "POM_5V_IN")


class Sampler:
    """Runs tegrastats in the background and keeps every reading."""

    def __init__(self, interval_ms=500):
        self.interval = interval_ms
        self.rows = []            # (time, phase, {rail: mW})
        self.phase = "start"
        self.proc = None
        self._t = None

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            raise SystemExit("tegrastats not found. This must run on the "
                             "Jetson itself.")
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        time.sleep(1.5)
        if not self.rows:
            raise SystemExit("tegrastats gave no output. Try sudo.")

    def _read(self):
        for line in self.proc.stdout:
            rails = {m.group(1): int(m.group(2)) for m in RAIL.finditer(line)}
            if rails:
                self.rows.append((time.time(), self.phase, rails))

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


def whole_board(rails):
    for k in WHOLE_BOARD:
        if k in rails:
            return k, rails[k]
    return "sum of rails", sum(rails.values())


def phase_values(rows, phase, rail=None):
    out = []
    for _t, ph, rails in rows:
        if ph != phase:
            continue
        if rail is None:
            out.append(whole_board(rails)[1])
        elif rail in rails:
            out.append(rails[rail])
    return out


def mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--cmd", required=True, help="the inference command")
    ap.add_argument("--windows", type=int, required=True,
                    help="windows scored by ONE run of the command")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--idle-seconds", type=int, default=60)
    ap.add_argument("--voltage", type=float, default=5.0)
    ap.add_argument("--period", type=float, default=30.0,
                    help="seconds between windows on a worn device")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    s = Sampler()
    s.start()
    which = whole_board(s.rows[-1][2])[0]
    print("reading the board from: %s" % which)
    print("power mode: ", end="", flush=True)
    subprocess.run("nvpmodel -q 2>/dev/null | tail -n 2 | tr '\\n' ' '; echo",
                   shell=True)

    def idle(label):
        s.phase = label
        print("\n%s: leave the board alone for %d s ..." % (label,
                                                           a.idle_seconds),
              flush=True)
        time.sleep(a.idle_seconds)

    idle("idle_before")

    s.phase = "running"
    print("\nrunning: %d x [%s]" % (a.repeat, a.cmd), flush=True)
    t0 = time.time()
    for i in range(a.repeat):
        print("  run %d of %d" % (i + 1, a.repeat), flush=True)
        r = subprocess.run(a.cmd, shell=True, stdout=subprocess.DEVNULL)
        if r.returncode != 0:
            s.stop()
            raise SystemExit("the command failed (exit %d). Run it by hand "
                             "first to see why." % r.returncode)
    run_s = time.time() - t0

    idle("idle_after")
    s.stop()

    # ---- numbers
    ib, ib_sd = mean_sd(phase_values(s.rows, "idle_before"))
    ia, ia_sd = mean_sd(phase_values(s.rows, "idle_after"))
    run, run_sd = mean_sd(phase_values(s.rows, "running"))
    if ib_sd <= ia_sd:
        idle_mw, idle_sd, pick = ib, ib_sd, "the first idle"
    else:
        idle_mw, idle_sd, pick = ia, ia_sd, "the second idle"
    p_inf = run - idle_mw

    total_windows = a.windows * a.repeat
    t_win = run_s / total_windows
    e_win = p_inf * t_win                                  # mW x s = mJ
    q_win = e_win / 1000.0 / a.voltage / 3.6               # mAh
    duty = t_win / a.period
    p_added = p_inf * duty
    mah_day = p_added / a.voltage * 24.0
    idle_day = idle_mw / a.voltage * 24.0

    L = []
    L.append("power source           : %s (whole board)" % which)
    L.append("samples                : idle_before %d, running %d, "
             "idle_after %d"
             % (len(phase_values(s.rows, "idle_before")),
                len(phase_values(s.rows, "running")),
                len(phase_values(s.rows, "idle_after"))))
    L.append("")
    L.append("%-14s %10s %10s" % ("", "mean (mW)", "spread (mW)"))
    L.append("%-14s %10.0f %10.0f" % ("idle, first", ib, ib_sd))
    L.append("%-14s %10.0f %10.0f" % ("running", run, run_sd))
    L.append("%-14s %10.0f %10.0f" % ("idle, second", ia, ia_sd))
    L.append("idle used            : %.0f mW  (%s, the steadier one)"
             % (idle_mw, pick))
    L.append("")
    L.append("INFERENCE POWER        : %.0f mW = %.2f W   "
             "(running - idle)" % (p_inf, p_inf / 1000.0))
    L.append("")
    L.append("by rail (mW)           %8s %8s %8s" % ("idle", "running",
                                                     "change"))
    rails = sorted({k for _t, _p, r in s.rows for k in r})
    for k in rails:
        i_v = phase_values(s.rows, "idle_before", k) + \
            phase_values(s.rows, "idle_after", k)
        r_v = phase_values(s.rows, "running", k)
        if i_v and r_v:
            L.append("  %-20s %8.0f %8.0f %+8.0f"
                     % (k, st.mean(i_v), st.mean(r_v),
                        st.mean(r_v) - st.mean(i_v)))
    L.append("")
    L.append("windows scored         : %d (%d per run x %d runs)"
             % (total_windows, a.windows, a.repeat))
    L.append("time per window        : %.3f s (includes starting python and "
             "loading the model each run)" % t_win)
    L.append("energy per window      : %.1f mJ" % e_win)
    L.append("charge per window      : %.4f mAh at %.1f V" % (q_win,
                                                              a.voltage))
    L.append("")
    L.append("on a worn device, one window every %.0f s:" % a.period)
    L.append("  busy                 : %.2f %% of the time" % (100 * duty))
    L.append("  added power          : %.0f mW  (%.1f mA at %.1f V)"
             % (p_added, p_added / a.voltage, a.voltage))
    L.append("  added per day        : %.0f mAh" % mah_day)
    L.append("  the board idling     : %.0f mAh per day" % idle_day)
    L.append("  inference share      : %.1f %% of the idle draw"
             % (100 * mah_day / idle_day if idle_day else float("nan")))
    L.append("")
    if abs(ib_sd - ia_sd) > 3 * min(ib_sd, ia_sd, 1e9) and max(ib_sd, ia_sd) > 300:
        L.append("NOTE: the two idle readings varied very differently "
                 "(%.0f vs %.0f mW). The steadier one was used." % (ib_sd,
                                                                    ia_sd))
    text = "\n".join(L)
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "power_summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(a.out, "power_trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unix_time", "phase"] + rails)
        for t, ph, r in s.rows:
            w.writerow([round(t, 2), ph] + [r.get(k, "") for k in rails])
    print("wrote power_summary.txt and power_trace.csv in %s" % a.out)


if __name__ == "__main__":
    main()

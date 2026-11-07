"""
add_resp_channel.py

Adds a seventh channel, respiration, to the mentor's make_windows_16hz.py,
and updates chronos_embed.py to expect seven channels instead of six.

Both originals are kept, with .orig appended, so nothing is lost.

    python3 add_resp_channel.py make_windows_16hz.py chronos_embed.py

WHY
    The AdaStress side of this project uses seven physiological channels.
    make_windows_16hz.py produces six: it reads respiratory_rate from the
    per-minute CSVs into the epoch array, but not as a 16 Hz channel. In our
    data that column is populated for about 1 percent of minutes, so it is
    not usable as a signal.

    The avro files do carry systolicPeaks, the time of every detected
    heartbeat, and breathing can be worked out from those.
"""
import re, shutil, sys

if len(sys.argv) < 2:
    sys.exit(__doc__)

windows_file = sys.argv[1]
embed_file = sys.argv[2] if len(sys.argv) > 2 else None

shutil.copy(windows_file, windows_file + ".orig")
src = open(windows_file).read()
n = 0
def sub(old, new):
    global src, n
    assert src.count(old) == 1, "FAIL(%d): %r" % (src.count(old), old[:70])
    src = src.replace(old, new); n += 1

# ---- 1. seven channels ----
sub('''CHANNELS = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp"]''',
'''CHANNELS = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp", "resp"]

# BTP-ADDED: respiration as a seventh channel.
#
# A wrist device does not measure breathing, but it leaves a trace in the
# heart rhythm: the gap between beats shortens on the in-breath and
# lengthens on the out-breath. The avro files carry systolicPeaks, which is
# the time of every detected beat, so the gaps can be worked out and
# band-passed to the breathing band.
#
# What goes in the channel is the WAVEFORM, not a rate. A rate of 15 breaths
# a minute repeated 480 times is a flat line, and a flat line gives Chronos
# nothing: we measured that on skin temperature, where 98 percent of values
# landed in a single bin. The band-passed interval series oscillates at the
# breathing frequency itself, which is real shape at 16 Hz.
#
# Use --resp-mode rate for the rate in breaths per minute instead.
#
# Worth knowing what this channel is worth. Checked against WESAD's chest
# respiration belt across all fifteen subjects, a rate derived this way
# correlated at 0.185. The level is about right but the moment to moment
# tracking is poor, because the modulation is weak in real wrist PPG.
RESP_LOW, RESP_HIGH = 0.1, 0.5      # 6 to 30 breaths a minute
RESP_GRID_HZ = 4.0                  # the gaps are resampled here before filtering
MIN_IBI, MAX_IBI = 0.333, 1.5       # 40 to 180 bpm; outside is a detection error''')

# ---- 2. read systolicPeaks from the avro ----
sub('''    parts = {c: [] for c in CHANNELS}''',
'''    parts = {c: [] for c in CHANNELS}
    peak_us = []        # BTP-ADDED: beat times, for the respiration channel''')

sub('''                for sensor, ch in (("bvp", "bvp"), ("eda", "eda"), ("temperature", "temp")):''',
'''                # BTP-ADDED: the heartbeat times. Different firmware spells
                # this field differently, so try the ones seen in practice.
                sp = raw.get("systolicPeaks")
                if sp:
                    for key in ("peaksTimeNanos", "peaksTimeNano",
                                "peakTimeNanos", "values"):
                        v = sp.get(key)
                        if v is not None and len(v):
                            peak_us.append(np.asarray(v, dtype=float) / 1e3)
                            break

                for sensor, ch in (("bvp", "bvp"), ("eda", "eda"), ("temperature", "temp")):''')

sub('''    out = {}
    for c, lst in parts.items():''',
'''    # BTP-ADDED: turn the beat times into a breathing channel
    if peak_us:
        beats = np.unique(np.concatenate(peak_us))
        print("[info] %d heartbeats found, for the respiration channel"
              % len(beats))
        parts["resp"] = [resp_from_peaks(beats)]
    else:
        print("[warn] no systolicPeaks in the avro files, so respiration "
              "will be empty")

    out = {}
    for c, lst in parts.items():''')

# ---- 3. the derivation ----
sub('''# ----------------------------------------------------------------------------- resample''',
'''def resp_from_peaks(beat_us, mode="wave"):
    """BTP-ADDED. A breathing channel from heartbeat times.

    beat_us is the time of every detected beat, in microseconds.

        gaps between beats
          -> drop anything outside 40 to 180 bpm, which is a detection error
          -> put the gaps on a steady 4 Hz grid
          -> band-pass 0.1 to 0.5 Hz, which is 6 to 30 breaths a minute
          -> that filtered series IS the breathing waveform

    Returned in the same shape as the other channels, so resample_to_grid
    can lift it to 16 Hz alongside them.

    mode="rate" returns breaths per minute instead, counted over a rolling
    30 seconds. Flatter, and therefore worse for Chronos, but it is what a
    person would recognise as a breathing rate.
    """
    empty = pd.DataFrame({"t": [], "v": [], "fs": []})
    b = np.sort(np.asarray(beat_us, dtype=float))
    if len(b) < 20:
        return empty

    sec = b / 1e6
    ibi = np.diff(sec)
    at = sec[1:]

    ok = (ibi >= MIN_IBI) & (ibi <= MAX_IBI)
    ibi, at = ibi[ok], at[ok]
    if len(ibi) < 20:
        return empty

    # Break where the sensor stopped reading. Interpolating a breathing
    # waveform across a gap of minutes would be inventing it.
    breaks = np.where(np.diff(at) > 30.0)[0]
    bounds = [0] + (breaks + 1).tolist() + [len(at)]

    nyq = RESP_GRID_HZ / 2.0
    sos = butter(2, [RESP_LOW / nyq, min(RESP_HIGH / nyq, 0.99)],
                 btype="band", output="sos")

    ts, vs = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg_at, seg_ibi = at[lo:hi], ibi[lo:hi]
        if len(seg_at) < 20 or seg_at[-1] - seg_at[0] < 30:
            continue
        grid = np.arange(seg_at[0], seg_at[-1], 1.0 / RESP_GRID_HZ)
        interp = np.interp(grid, seg_at, seg_ibi)
        filt = sosfiltfilt(sos, interp)

        if mode == "rate":
            # peaks per 30 s, carried along the grid
            from scipy.signal import find_peaks
            win = int(30 * RESP_GRID_HZ)
            rate = np.full(len(filt), np.nan)
            for s in range(0, len(filt) - win, int(RESP_GRID_HZ)):
                pk, _ = find_peaks(filt[s:s + win],
                                   distance=RESP_GRID_HZ * 1.5)
                rate[s + win // 2] = len(pk) * 2.0
            good = np.isfinite(rate)
            if good.sum() < 2:
                continue
            filt = np.interp(grid, grid[good], rate[good])

        ts.append(grid * 1e6)
        vs.append(filt)

    if not ts:
        return empty
    return pd.DataFrame({"t": np.concatenate(ts), "v": np.concatenate(vs),
                         "fs": RESP_GRID_HZ})


# ----------------------------------------------------------------------------- resample''')

# ---- 4. the flag, threaded through ----
sub('''    ap.add_argument("--minute-first-only", action="store_true",''',
'''    ap.add_argument("--resp-mode", choices=["wave", "rate"], default="wave",
                    help="what the respiration channel holds. wave is the "
                         "band-passed beat-interval series, which oscillates "
                         "at the breathing frequency and so has shape for "
                         "Chronos to read. rate is breaths per minute, which "
                         "is flatter and gives a tokeniser almost nothing.")
    ap.add_argument("--minute-first-only", action="store_true",''')

sub('''    data = read_avro_folder(a.raw, s_ts, e_ts)''',
'''    data = read_avro_folder(a.raw, s_ts, e_ts, a.resp_mode)''')

sub('''def read_avro_folder(raw_dir, start=None, end=None):''',
'''def read_avro_folder(raw_dir, start=None, end=None, resp_mode="wave"):''')

sub('''        parts["resp"] = [resp_from_peaks(beats)]''',
'''        parts["resp"] = [resp_from_peaks(beats, resp_mode)]''')

open(windows_file, "w").write(src)
print("%s: %d edits  (original kept as %s.orig)"
      % (windows_file, n, windows_file))

# ---- chronos_embed.py just needs to know there are seven ----
if embed_file:
    shutil.copy(embed_file, embed_file + ".orig")
    e = open(embed_file).read()
    old = '"""CH = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp"]"""'.strip('"')
    new = ('CH = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp", "resp"]\n'
           "\n"
           "# resp is added by add_resp_channel.py. It is the band-passed\n"
           "# beat-interval series, which oscillates at the breathing\n"
           "# frequency, rather than a rate in breaths per minute. A rate\n"
           "# repeated 480 times is a flat line, and a flat line gives the\n"
           "# tokeniser nothing to work with.")
    if old in e:
        e = e.replace(old, new)
        open(embed_file, "w").write(e)
        print("%s: channel list now has seven  (original kept as %s.orig)"
              % (embed_file, embed_file))
    else:
        print("%s: could not find the channel list; add \"resp\" to CH by "
              "hand" % embed_file)

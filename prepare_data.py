"""Convert raw phone exports into the project's labelled CSV format.

Every output file has the columns

    timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity

at the phone's NATIVE sampling rate (harmonising rates happens in the notebook,
so the per-member rates can be reported there).

  timestamp  seconds (Unix epoch when the app provides it, else elapsed)
  acc_*      m/s^2, INCLUDING gravity
  gyro_*     rad/s

Supported inputs
  * Sensor Logger (iOS/Android) export: a .zip or an unzipped folder with
    Accelerometer.csv / TotalAcceleration.csv / Gravity.csv / Gyroscope.csv
  * Physics Toolbox Sensor Suite CSV (gFx, gFy, gFz, wx, wy, wz)
  * A CSV that already uses the column names above

Batch mode (default) - drop exports into

    data/raw_exports/<split>/<member>/<activity>/<anything>.zip|folder|.csv

where <split> is train or test and <activity> is one of standing, walking,
jumping, still - or "mixed" for a continuous test recording that contains
several activities. A mixed recording needs a sidecar text file with the same
name plus ".segments.txt", one segment per line, times in seconds from the
start of the recording:

    standing 0 10
    walking 10 20
    jumping 20 30
    still 30 40

Any time not covered by a segment is labelled "transition".

Then run:  python prepare_data.py
Output:    data/<split>/<member>_<activity>_<NN>.csv
(data/train and data/test are rebuilt from raw_exports on every run.)
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITIES = ["standing", "walking", "jumping", "still"]
G = 9.80665
OUT_COLS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "activity"]


# ---------------------------------------------------------------- readers
def _read_sensor_logger(files: dict[str, bytes]) -> pd.DataFrame:
    """files maps lower-cased file name -> raw bytes."""

    def load(name):
        raw = files.get(name.lower() + ".csv")
        return None if raw is None else pd.read_csv(io.BytesIO(raw))

    total, acc, grav, gyro = (load(n) for n in
                              ("TotalAcceleration", "Accelerometer", "Gravity", "Gyroscope"))
    if gyro is None or (total is None and acc is None):
        raise ValueError("Sensor Logger export needs Gyroscope.csv and an accelerometer file")

    # Sensor Logger's "Accelerometer" has gravity removed; we want it included,
    # because gravity direction is what separates "still on a table" from
    # "standing with the phone at the waist".
    if total is not None:
        acc_df = total
    elif grav is not None:
        acc_df = acc.copy()
        for ax in "xyz":
            acc_df[ax] = acc[ax].to_numpy() + np.interp(acc["time"], grav["time"], grav[ax])
    else:
        print("   ! no TotalAcceleration/Gravity file: accelerometer has gravity removed. "
              "Enable 'Total Acceleration' or 'Gravity' in Sensor Logger.", file=sys.stderr)
        acc_df = acc

    t_acc = acc_df["time"].to_numpy(dtype=float) / 1e9   # ns epoch -> s
    t_gyr = gyro["time"].to_numpy(dtype=float) / 1e9
    out = pd.DataFrame({"timestamp": t_acc})
    for ax in "xyz":
        out[f"acc_{ax}"] = acc_df[ax].to_numpy(dtype=float)
        out[f"gyro_{ax}"] = np.interp(t_acc, t_gyr, gyro[ax].to_numpy(dtype=float))
    return out


def _read_csv_any(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.strip().lower(): c for c in df.columns}
    if all(c in cols for c in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]):
        tcol = cols.get("timestamp") or cols.get("time")
        out = df.rename(columns={v: k for k, v in cols.items()})
        out["timestamp"] = pd.to_numeric(df[tcol], errors="coerce")
        return out[OUT_COLS[:-1]]
    if all(c in cols for c in ["gfx", "gfy", "gfz", "wx", "wy", "wz"]):   # Physics Toolbox
        t = pd.to_numeric(df[cols["time"]], errors="coerce")
        if t.isna().all():
            raise ValueError("Physics Toolbox time column is not numeric; export with time in seconds")
        return pd.DataFrame({
            "timestamp": t,
            "acc_x": df[cols["gfx"]] * G, "acc_y": df[cols["gfy"]] * G, "acc_z": df[cols["gfz"]] * G,
            "gyro_x": df[cols["wx"]], "gyro_y": df[cols["wy"]], "gyro_z": df[cols["wz"]],
        })
    raise ValueError(f"unrecognised CSV columns: {list(df.columns)[:12]}")


def read_recording(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = {p.name.lower(): p.read_bytes() for p in path.glob("*.csv")}
        df = _read_sensor_logger(files)
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            files = {Path(n).name.lower(): z.read(n) for n in z.namelist() if n.lower().endswith(".csv")}
        df = _read_sensor_logger(files)
    else:
        df = _read_csv_any(pd.read_csv(path))

    df = (df.dropna().sort_values("timestamp")
            .drop_duplicates("timestamp").reset_index(drop=True))
    # Units guard: total acceleration should have magnitude ~9.8 m/s^2.
    # If it averages ~1 the app exported in g.
    mag = np.sqrt((df[["acc_x", "acc_y", "acc_z"]] ** 2).sum(axis=1)).median()
    if 0.5 < mag < 2.0:
        df[["acc_x", "acc_y", "acc_z"]] *= G
    return df


def label_mixed(df: pd.DataFrame, seg_file: Path) -> pd.DataFrame:
    t = df["timestamp"] - df["timestamp"].iloc[0]
    # Seconds outside every segment (moving between activities, putting the
    # phone down) stay in the file so the recording remains continuous; they
    # are labelled "transition" and left out of the evaluation.
    df["activity"] = "transition"
    for line in seg_file.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        act, start, end = line.split()
        if act not in ACTIVITIES:
            raise ValueError(f"{seg_file.name}: unknown activity '{act}'")
        df.loc[(t >= float(start)) & (t < float(end)), "activity"] = act
    return df


# ---------------------------------------------------------------- batch
def next_index(out_dir: Path, stem: str) -> int:
    used = [int(p.stem.rsplit("_", 1)[1]) for p in out_dir.glob(f"{stem}_*.csv")
            if p.stem.rsplit("_", 1)[1].isdigit()]
    return max(used, default=0) + 1


def convert(src: Path, split: str, member: str, activity: str, out_root: Path) -> Path:
    df = read_recording(src)
    if activity == "mixed":
        seg = src.parent / (src.name.removesuffix(".zip").removesuffix(".csv") + ".segments.txt")
        if not seg.exists():
            raise FileNotFoundError(f"mixed recording needs {seg.name}")
        df = label_mixed(df, seg)
    else:
        df["activity"] = activity
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{member.lower()}_{activity}"
    out = out_dir / f"{stem}_{next_index(out_dir, stem):02d}.csv"
    df[OUT_COLS].to_csv(out, index=False, float_format="%.6f")
    dur = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    fs = (len(df) - 1) / dur if dur > 0 else float("nan")
    print(f"   {src.name} -> {out.name}  ({dur:.1f} s, {fs:.0f} Hz)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data", help="data directory (default: data)")
    args = ap.parse_args()
    root = Path(args.root)
    exports = root / "raw_exports"
    if not exports.exists():
        sys.exit(f"nothing to do: {exports} does not exist")

    # raw_exports is the source of truth: rebuild the CSV folders from scratch
    for split in ("train", "test"):
        for old in (root / split).glob("*.csv"):
            old.unlink()

    n_ok = n_err = 0
    for split_dir in sorted(p for p in exports.iterdir() if p.is_dir()):
        for member_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for act_dir in sorted(p for p in member_dir.iterdir() if p.is_dir()):
                act = act_dir.name.lower()
                if act not in ACTIVITIES + ["mixed"]:
                    print(f"skip folder {act_dir} (not an activity name)")
                    continue
                print(f"{split_dir.name}/{member_dir.name}/{act}")
                for src in sorted(act_dir.iterdir()):
                    if src.name.endswith(".segments.txt"):
                        continue
                    try:
                        convert(src, split_dir.name, member_dir.name, act, root)
                        n_ok += 1
                    except Exception as e:  # keep going, report at the end
                        print(f"   ! {src.name}: {e}", file=sys.stderr)
                        n_err += 1
    print(f"\nconverted {n_ok} recording(s), {n_err} error(s)")


if __name__ == "__main__":
    main()

# Human Activity Recognition with a Hidden Markov Model

This project infers whether a person is **standing, walking, jumping or still** from smartphone accelerometer and gyroscope data. It uses a Gaussian Hidden Markov Model implemented from scratch in numpy, covering forward–backward, Baum–Welch and Viterbi.

## Repository layout

```
data/
  raw_exports/       Sensor Logger zips as exported (input to prepare_data.py)
  train/             labelled training CSVs: <name>_<activity>_<NN>.csv
  test/              unseen test CSVs (never used for training)
prepare_data.py      converts app exports into the CSV format below
hmm_activity.ipynb   the full pipeline: preprocessing, features, HMM, evaluation
report/
  figures/           figures saved by the notebook
```

Every CSV has the columns `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity`:
- `timestamp` is in seconds
- acceleration is in m/s² and includes gravity
- the gyroscope is in rad/s
- each file is stored at the phone's native sampling rate (the notebook resamples to 50 Hz)

## Recording protocol

1. In **Sensor Logger**, enable **Accelerometer**, **Gravity** and **Gyroscope**.
2. Record the training clips, 5–10 s each, about 13 per activity, so that each activity totals at least 90 s:
   - **standing**: phone held steady at waist level
   - **walking**: steady pace, phone at waist level
   - **jumping**: continuous jumps, phone at waist level
   - **still**: phone flat on a table
3. Record **2 or more test sessions** on a different day or in a different place. In each one, do all four activities back to back for about 10 s each, using a stopwatch, and note the order.
4. Export each recording as a zip.
5. Record one ~5 s video per activity showing your face and full body, phone in hand, in good light.

## Running

```bash
pip install -r requirements.txt
```

Put the exports into folders named by split, member and activity:

```
data/raw_exports/train/<name>/<activity>/<export>.zip     # activity = standing|walking|jumping|still
data/raw_exports/test/<name>/mixed/<export>.zip           # continuous test session
data/raw_exports/test/<name>/mixed/<export>.segments.txt  # "walking 10 20" per line (seconds)
```

Then run:

```bash
python prepare_data.py    # rebuilds data/train and data/test
jupyter notebook hmm_activity.ipynb
```

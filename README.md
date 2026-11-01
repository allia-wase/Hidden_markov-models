# Human Activity Recognition with a Hidden Markov Model

I recorded my own phone data for four activities (standing, walking, jumping and still) and built a Gaussian Hidden Markov Model from scratch in numpy to recognise them. The implementation includes forward-backward, Baum-Welch and Viterbi. On three unseen test sessions the model labels 92.6% of windows correctly.

## Repository layout

```
data/
  train/             61 labelled training clips: alliane_<activity>_<NN>.csv
  test/              3 continuous test sessions: alliane_mixed_<NN>.csv
prepare_data.py      converts Sensor Logger exports into the CSV format below
hmm_activity.ipynb   the full pipeline: preprocessing, features, HMM, evaluation
report/
  figures/           figures saved by the notebook
```

## Dataset

All data was recorded on an iPhone 14 with Sensor Logger at 100 Hz. For standing, walking and jumping I held the phone flat with the screen up at waist level; for still it lay flat on a table.

| Activity | Training clips | Training time | Test windows |
|---|---|---|---|
| Standing | 15 | 103.7 s | 29 |
| Walking | 16 | 118.9 s | 33 |
| Jumping | 15 | 98.1 s | 30 |
| Still | 15 | 99.5 s | 29 |

### Data collection videos

| Activity | Video |
|---|---|
| Standing | [Watch](https://drive.google.com/file/d/1Hc-7JuFMNjDIyc8vTTDgJF645VrOVo9M/view?usp=drive_link) |
| Walking | [Watch](https://drive.google.com/file/d/1VTWMarIPIKNewofSt5uE4qYsZ0iSnmiz/view?usp=drive_link) |
| Jumping | [Watch](https://drive.google.com/file/d/1depReIVwLN4IAf_veE4kjWD8ZHYJcCQl/view?usp=drive_link) |
| Still | [Watch](https://drive.google.com/file/d/1E6PHQBcCtrlErBuN93QFxcTyeGweL2In/view?usp=drive_link) |

Every training clip is 5 to 10 s long and contains one activity. Each test session is a single recording of 44 to 46 s with all four activities back to back in a different order. The seconds spent moving between activities are labelled `transition` and are not scored.

Every CSV has the columns `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity`:
- `timestamp` is in seconds (Unix time)
- acceleration is in m/s² and includes gravity
- the gyroscope is in rad/s
- files are stored at the native 100 Hz; the notebook resamples them to 50 Hz

## Results on unseen data

| Activity | Sensitivity | Specificity |
|---|---|---|
| Standing | 0.759 | 1.000 |
| Walking | 0.970 | 0.909 |
| Jumping | 1.000 | 0.989 |
| Still | 0.966 | 1.000 |

Overall accuracy is 0.926. Every error falls at a change of activity, where a 2 s window covers the end of one activity and the start of the next.

## Running

```bash
pip install -r requirements.txt
jupyter notebook hmm_activity.ipynb
```

The notebook reads the CSVs in `data/` directly. To rebuild them from new Sensor Logger exports, put the zips in `data/raw_exports/` and run `python prepare_data.py`. That folder is not committed, and the docstring at the top of `prepare_data.py` explains how to lay it out.

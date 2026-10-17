# RailSense

Train delay predictor and connection-risk advisor. Enter your train, the weather and the gap
to your next train, and RailSense predicts the delay and the chance you miss the connection.

## Project layout

```
railsense/
├── .streamlit/config.toml   theme colours
├── data/                    trains.csv (made by make_data.py)
├── models/                  railsense.joblib (made by train.py)
├── app.py                   Streamlit interface
├── railsense_core.py        prediction logic, waterfall, gauge, report
├── make_data.py             creates the simulated dataset
├── train.py                 compares 3 models and saves the best
└── requirements.txt
```

## Run it

```
python make_data.py
python train.py
streamlit run app.py
```

Run `train.py` again whenever the data or the model changes.

## How it works

1. A regression model predicts the delay in minutes from seven inputs.
2. The spread of the model's errors (sigma) turns that prediction into a probability:
   chance of missing = P(delay > gap between trains).
3. The app shows that risk as a gauge, a curve, a what-if analysis, a delay breakdown,
   a best-time scan and a comparison between two trains.

## Data

The data is **simulated** (8,000 trips from a formula in `make_data.py`), so the model
mostly learns that formula back. To use real records, save a CSV at `data/trains.csv`
with the same columns (`train_type, hour, day_of_week, month, distance_km, weather,
prev_delay_min, delay_min`) and run `train.py` again. Nothing else needs to change.

## Deploy free on Streamlit Community Cloud

1. Make sure `requirements.txt` lists `streamlit`, `pandas`, `numpy`, `scipy`, `joblib`
   and `scikit-learn` pinned to the version you trained with (`pip show scikit-learn`).
2. Commit these to a GitHub repository: `app.py`, `railsense_core.py`, `requirements.txt`,
   `.streamlit/config.toml` and the `models/` folder. Do not commit `venv/`.
3. Go to share.streamlit.io, sign in with GitHub, choose the repository, set the main
   file to `app.py` and click Deploy.
4. Add screenshots to this README once it is live.
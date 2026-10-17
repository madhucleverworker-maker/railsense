import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 8000

df = pd.DataFrame({
    "train_type": rng.choice(["Express", "Superfast", "Passenger", "Rajdhani"], n),
    "hour": rng.integers(0, 24, n),
    "day_of_week": rng.integers(0, 7, n),
    "month": rng.integers(1, 13, n),
    "distance_km": rng.integers(50, 2200, n),
    "weather": rng.choice(["Clear", "Rain", "Fog"], n, p=[0.6, 0.25, 0.15]),
    "prev_delay_min": rng.integers(0, 90, n),
})

delay = (
    0.35 * df.prev_delay_min
    + 0.01 * df.distance_km
    + np.where(df.weather == "Fog", 25, np.where(df.weather == "Rain", 10, 0))
    + np.where(df.train_type == "Passenger", 12, 0)
    + np.where(df.month.isin([12, 1]), 15, 0)
    + np.where(df.hour.between(17, 21), 8, 0)
    + rng.normal(0, 8, n)
)
df["delay_min"] = delay.clip(lower=0).round()
df.to_csv("data/trains.csv", index=False)
print("Saved", df.shape)
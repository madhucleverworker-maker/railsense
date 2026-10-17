import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/trains.csv")
X, y = df.drop(columns="delay_min"), df["delay_min"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

CAT = ["train_type", "weather"]


def build(regressor):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("reg", regressor)])


candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

rows, fitted, preds = [], {}, {}
for name, reg in candidates.items():
    pipe = build(reg).fit(X_tr, y_tr)
    pred = np.clip(pipe.predict(X_te), 0, None)  # the app also clips at 0
    fitted[name], preds[name] = pipe, pred
    rows.append(
        {
            "Model": name,
            "MAE (min)": round(mean_absolute_error(y_te, pred), 2),
            "R²": round(r2_score(y_te, pred), 3),
        }
    )
    print(f"{name:20s} MAE {rows[-1]['MAE (min)']:.2f} min | R² {rows[-1]['R²']:.3f}")

comparison = pd.DataFrame(rows)
best_name = comparison.loc[comparison["MAE (min)"].idxmin(), "Model"]
model, pred = fitted[best_name], preds[best_name]

mae = float(mean_absolute_error(y_te, pred))
sigma = float(np.std(y_te - pred))  # used for connection risk
print(f"\nBest: {best_name} | MAE {mae:.1f} min | residual std {sigma:.1f} min")

# Which inputs matter most: scramble one input at a time and see how much the error grows.
perm = permutation_importance(
    model, X_te, y_te, n_repeats=5, random_state=42,
    scoring="neg_mean_absolute_error",
)
importance = dict(zip(X_te.columns, perm.importances_mean.round(3)))

# A sample of test trips for the predicted-vs-actual chart.
sample = np.random.default_rng(0).choice(len(y_te), size=min(600, len(y_te)), replace=False)

joblib.dump(
    {
        "model": model,
        "model_name": best_name,
        "sigma": sigma,
        "mae": mae,
        "comparison": comparison,
        "importance": importance,
        "test_actual": y_te.to_numpy()[sample],
        "test_pred": pred[sample],
    },
    "models/railsense.joblib",
)
print("Saved models/railsense.joblib")
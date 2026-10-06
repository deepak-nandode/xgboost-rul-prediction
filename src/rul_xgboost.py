"""
Remaining Useful Life (RUL) prediction on NASA C-MAPSS FD001 using XGBoost.

Pipeline:
  1. Load data and build the RUL label (capped at 125 cycles)
  2. Drop constant columns, scale sensors
  3. Add rolling-mean features (window = 10 cycles, per engine)
  4. Split train/validation BY ENGINE (no leakage)
  5. Linear Regression baseline vs XGBoost
  6. Hand-tune max_depth, learning_rate, n_estimators on validation set
  7. Retrain on all training engines, evaluate on the official test set
  8. Save plots to outputs/

Run from the project root:  python src/rul_xgboost.py
"""
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RUL_CAP = 125      # common practice: early in life, degradation is not visible
WINDOW = 10        # rolling-mean window (cycles)
SEED = 42

COLS = (["engine", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)])


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------- 1. load
def load_data():
    train = pd.read_csv(DATA / "train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(DATA / "test_FD001.txt", sep=r"\s+", header=None, names=COLS)
    rul_test = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul_test


def add_rul(train):
    """RUL = (last cycle of the engine) - (current cycle), capped."""
    max_cycle = train.groupby("engine")["cycle"].transform("max")
    train["RUL"] = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)
    return train


# ---------------------------------------------------------- 2. preprocess
def find_constant_cols(train):
    """Sensors/settings that (almost) never change carry no information."""
    cols = [c for c in COLS if c.startswith(("op", "s"))]
    return [c for c in cols if train[c].std() < 1e-6 or c == "op3"]


def scale(train, test, feature_cols):
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])  # fit on train only
    return train, test


# ----------------------------------------------------------- 3. features
def add_rolling_mean(df, sensor_cols):
    """Per-engine rolling mean: smooths noise and shows the trend."""
    rolled = (df.groupby("engine")[sensor_cols]
                .rolling(WINDOW, min_periods=1).mean()
                .reset_index(level=0, drop=True))
    rolled.columns = [f"{c}_roll{WINDOW}" for c in sensor_cols]
    return pd.concat([df, rolled], axis=1)


# --------------------------------------------------------------- plots
def plot_sensors(train_raw, sensors):
    fig, axes = plt.subplots(len(sensors), 1, figsize=(8, 2.2 * len(sensors)), sharex=True)
    for ax, s in zip(axes, sensors):
        for eng in [1, 2, 3, 4, 5]:
            d = train_raw[train_raw.engine == eng]
            ax.plot(d["cycle"], d[s], alpha=0.8)
        ax.set_ylabel(s)
    axes[-1].set_xlabel("cycle")
    fig.suptitle("Sensor readings over time for 5 engines")
    fig.tight_layout()
    fig.savefig(OUT / "01_sensor_trends.png", dpi=130)
    plt.close(fig)


def plot_pred(y_true, y_pred, title, fname):
    order = np.argsort(y_true)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(y_true, y_pred, alpha=0.7)
    lim = [0, max(y_true.max(), y_pred.max()) + 5]
    ax[0].plot(lim, lim, "r--", label="perfect prediction")
    ax[0].set_xlabel("Actual RUL"); ax[0].set_ylabel("Predicted RUL")
    ax[0].legend(); ax[0].set_title(title)
    ax[1].plot(np.asarray(y_true)[order], label="Actual")
    ax[1].plot(np.asarray(y_pred)[order], label="Predicted")
    ax[1].set_xlabel("Test engines (sorted by actual RUL)"); ax[1].set_ylabel("RUL")
    ax[1].legend(); ax[1].set_title("Sorted view")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def plot_importance(model, feature_names, top=15):
    imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax)
    ax.set_title(f"XGBoost feature importance (top {top})")
    fig.tight_layout()
    fig.savefig(OUT / "03_feature_importance.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    train, test, rul_test = load_data()
    train = add_rul(train)
    print(f"Train: {train.shape}, engines: {train.engine.nunique()}")
    print(f"Test : {test.shape}, engines: {test.engine.nunique()}")

    plot_sensors(train.copy(), ["s2", "s7", "s11", "s12"])

    drop_cols = find_constant_cols(train)
    sensor_cols = [c for c in COLS if c.startswith(("op", "s")) and c not in drop_cols]
    print(f"Dropped constant columns: {drop_cols}")
    print(f"Kept {len(sensor_cols)} columns: {sensor_cols}")

    train, test = scale(train, test, sensor_cols)
    train = add_rolling_mean(train, sensor_cols)
    test = add_rolling_mean(test, sensor_cols)
    feature_cols = sensor_cols + [f"{c}_roll{WINDOW}" for c in sensor_cols]

    # ---- split by engine ID (never by row)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(train, groups=train["engine"]))
    tr, val = train.iloc[tr_idx], train.iloc[val_idx]
    X_tr, y_tr = tr[feature_cols], tr["RUL"]
    X_val, y_val = val[feature_cols], val["RUL"]
    print(f"Train engines: {tr.engine.nunique()}, validation engines: {val.engine.nunique()}")

    # ---- baseline
    lr = LinearRegression().fit(X_tr, y_tr)
    lr_val = rmse(y_val, lr.predict(X_val))
    print(f"\n[Validation] Linear Regression RMSE: {lr_val:.2f}")

    # ---- XGBoost default
    xgb_default = XGBRegressor(random_state=SEED).fit(X_tr, y_tr)
    print(f"[Validation] XGBoost (default)  RMSE: {rmse(y_val, xgb_default.predict(X_val)):.2f}")

    # ---- manual tuning of 3 parameters
    grid = {"max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 400]}
    results = []
    for md, lrate, n in product(grid["max_depth"], grid["learning_rate"], grid["n_estimators"]):
        m = XGBRegressor(max_depth=md, learning_rate=lrate, n_estimators=n,
                         random_state=SEED).fit(X_tr, y_tr)
        results.append({"max_depth": md, "learning_rate": lrate, "n_estimators": n,
                        "val_rmse": rmse(y_val, m.predict(X_val))})
    res = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)
    res.to_csv(OUT / "tuning_results.csv", index=False)
    print("\nTuning results (top 5):")
    print(res.head(5).to_string(index=False))
    best = res.iloc[0]

    # ---- final model: retrain on ALL training engines, test on official test set
    full_X, full_y = train[feature_cols], train["RUL"]
    final_lr = LinearRegression().fit(full_X, full_y)
    final_xgb = XGBRegressor(max_depth=int(best.max_depth),
                             learning_rate=float(best.learning_rate),
                             n_estimators=int(best.n_estimators),
                             random_state=SEED).fit(full_X, full_y)

    # test set: predict from the LAST recorded cycle of each engine
    last = test.groupby("engine").tail(1).reset_index(drop=True)
    y_test = rul_test["RUL"].clip(upper=RUL_CAP).values
    pred_lr = final_lr.predict(last[feature_cols])
    pred_xgb = np.clip(final_xgb.predict(last[feature_cols]), 0, RUL_CAP)

    lr_test, xgb_test = rmse(y_test, pred_lr), rmse(y_test, pred_xgb)
    print("\n=== FINAL TEST RESULTS (100 test engines) ===")
    print(f"Linear Regression RMSE: {lr_test:.2f}")
    print(f"XGBoost           RMSE: {xgb_test:.2f}")
    print(f"Improvement: {100 * (lr_test - xgb_test) / lr_test:.1f}%")

    pd.DataFrame({"model": ["Linear Regression", "XGBoost (tuned)"],
                  "test_rmse": [lr_test, xgb_test]}).to_csv(OUT / "final_results.csv", index=False)

    plot_pred(y_test, pred_xgb, f"XGBoost: actual vs predicted (RMSE {xgb_test:.1f})",
              "02_actual_vs_predicted.png")
    plot_importance(final_xgb, feature_cols)
    print(f"\nPlots and tables saved in: {OUT}")


if __name__ == "__main__":
    main()

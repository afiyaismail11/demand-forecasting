
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "sales_data.csv")


def make_sample_data():
    rng = np.random.default_rng(42)
    months = pd.date_range("2023-11-01", periods=36, freq="MS")
    trend = np.linspace(200, 320, 36)
    season = np.array([0, -10, -5, 5, 10, 0, -15, -10, 5, 30, 45, 35])
    seasonal = season[months.month.values - 1]
    noise = rng.normal(0, 8, 36)
    units = (trend + seasonal + noise).round().astype(int)
    pd.DataFrame({"month": months, "units_sold": units}).to_csv(DATA_FILE, index=False)


def design_matrix(t, month_numbers):
    cols = [np.ones(len(t)), t]
    for m in range(2, 13):
        cols.append((month_numbers == m).astype(float))
    return np.column_stack(cols)


def fit_predict(train_t, train_m, train_y, new_t, new_m):
    X = design_matrix(train_t, train_m)
    coef, *_ = np.linalg.lstsq(X, train_y, rcond=None)
    return design_matrix(new_t, new_m) @ coef


def mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100


def main():
    if not os.path.exists(DATA_FILE):
        make_sample_data()

    df = pd.read_csv(DATA_FILE, parse_dates=["month"])
    t = np.arange(len(df))
    m = df["month"].dt.month.values
    y = df["units_sold"].values.astype(float)

    split = len(df) - 6
    test_pred = fit_predict(t[:split], m[:split], y[:split], t[split:], m[split:])
    print(f"Model error (MAPE): {mape(y[split:], test_pred):.1f}%")

    future_months = pd.date_range(
        df["month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS"
    )
    future_t = np.arange(len(df), len(df) + 6)
    future_pred = fit_predict(t, m, y, future_t, future_months.month.values)

    result = pd.DataFrame({
        "month": future_months.strftime("%b %Y"),
        "forecast_units": future_pred.round().astype(int),
    })
    print(result.to_string(index=False))
    result.to_csv(os.path.join(BASE, "forecast_result.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["month"], y, marker="o", label="Actual sales")
    plt.plot(future_months, future_pred, marker="o", linestyle="--", color="red", label="Forecast")
    plt.title("Demand Forecasting - Monthly Sales")
    plt.xlabel("Month")
    plt.ylabel("Units sold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "forecast_chart.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Problem 1: 光伏电站发电功率日前预测

使用方法（示例）:
  python problem1_solution.py --data /path/to/data.csv --output-dir outputs

数据要求（至少包含以下字段中的一个别名）:
  - 时间列: timestamp / time / datetime / date_time
  - 功率列: power / pv_power / output / generation
  - 天气列（可选）: irradiance / ghi / temperature / cloud_cover / humidity / wind_speed
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TIME_CANDIDATES = ["timestamp", "time", "datetime", "date_time", "date"]
POWER_CANDIDATES = ["power", "pv_power", "output", "generation", "p"]
IRRADIANCE_CANDIDATES = ["irradiance", "ghi", "solar_irradiance", "radiation"]
TEMPERATURE_CANDIDATES = ["temperature", "temp", "air_temperature"]
CLOUD_CANDIDATES = ["cloud_cover", "cloud", "clouds", "cloudiness"]
HUMIDITY_CANDIDATES = ["humidity", "rh"]
WIND_CANDIDATES = ["wind_speed", "wind"]


@dataclass(frozen=True)
class ColumnMap:
    time: str
    power: str
    irradiance: Optional[str]
    temperature: Optional[str]
    cloud_cover: Optional[str]
    humidity: Optional[str]
    wind_speed: Optional[str]

    @property
    def weather_columns(self) -> List[str]:
        columns = [
            self.irradiance,
            self.temperature,
            self.cloud_cover,
            self.humidity,
            self.wind_speed,
        ]
        return [col for col in columns if col is not None]


def _infer_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def infer_column_map(df: pd.DataFrame) -> ColumnMap:
    time_col = _infer_column(df.columns, TIME_CANDIDATES)
    power_col = _infer_column(df.columns, POWER_CANDIDATES)
    if not time_col or not power_col:
        raise ValueError(
            "缺少必要列。请确保包含时间列和功率列（可在脚本开头查看可接受的列名）。"
        )
    return ColumnMap(
        time=time_col,
        power=power_col,
        irradiance=_infer_column(df.columns, IRRADIANCE_CANDIDATES),
        temperature=_infer_column(df.columns, TEMPERATURE_CANDIDATES),
        cloud_cover=_infer_column(df.columns, CLOUD_CANDIDATES),
        humidity=_infer_column(df.columns, HUMIDITY_CANDIDATES),
        wind_speed=_infer_column(df.columns, WIND_CANDIDATES),
    )


def load_and_prepare(path: Path) -> Tuple[pd.DataFrame, ColumnMap]:
    df = pd.read_csv(path)
    column_map = infer_column_map(df)
    df[column_map.time] = pd.to_datetime(df[column_map.time], errors="coerce")
    df = df.dropna(subset=[column_map.time])
    df = df.sort_values(column_map.time)
    df = df.set_index(column_map.time)

    numeric_columns = [column_map.power] + column_map.weather_columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df = df.resample("15min").mean()
    df = df.interpolate(limit_direction="both")
    return df, column_map


def build_test_windows(index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    years = sorted(index.year.unique())
    for year in years:
        for month in (2, 5, 8, 11):
            month_mask = (index.year == year) & (index.month == month)
            if not month_mask.any():
                continue
            last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            start_day = last_day - pd.Timedelta(days=6)
            start = pd.Timestamp(start_day.date())
            end = start + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)
            window_mask = (index >= start) & (index <= end)
            if not window_mask.any():
                continue
            windows.append((start, end))
    return windows


def daylight_mask(df: pd.DataFrame, column_map: ColumnMap) -> pd.Series:
    if column_map.irradiance and column_map.irradiance in df.columns:
        return df[column_map.irradiance] > 0.01
    if column_map.power in df.columns:
        return df[column_map.power] > 0.01
    hours = df.index.hour + df.index.minute / 60
    return (hours >= 6) & (hours <= 18)


def baseline_forecast(
    df: pd.DataFrame,
    column_map: ColumnMap,
    test_index: pd.DatetimeIndex,
    baseline_days: int,
    fallback_by_time: Dict[time, float],
) -> pd.Series:
    predictions = []
    for ts in test_index:
        history = []
        for offset in range(1, baseline_days + 1):
            prev_ts = ts - pd.Timedelta(days=offset)
            if prev_ts in df.index:
                history.append(df.at[prev_ts, column_map.power])
        if history:
            predictions.append(float(np.nanmean(history)))
        else:
            predictions.append(float(fallback_by_time.get(ts.time(), np.nan)))
    return pd.Series(predictions, index=test_index)


def time_of_day_mean(series: pd.Series) -> Dict[time, float]:
    grouped = series.groupby(series.index.time).mean()
    return {time_key: value for time_key, value in grouped.items()}


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hours = index.hour + index.minute / 60
    day_of_year = index.dayofyear
    two_pi = 2 * np.pi
    return pd.DataFrame(
        {
            "hour_sin": np.sin(two_pi * hours / 24),
            "hour_cos": np.cos(two_pi * hours / 24),
            "doy_sin": np.sin(two_pi * day_of_year / 365.25),
            "doy_cos": np.cos(two_pi * day_of_year / 365.25),
        },
        index=index,
    )


def build_weather_features(df: pd.DataFrame, column_map: ColumnMap) -> pd.DataFrame:
    features = time_features(df.index)
    if column_map.weather_columns:
        features = pd.concat([df[column_map.weather_columns], features], axis=1)
    return features


def evaluate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    valid = ~(y_true.isna() | y_pred.isna())
    if valid.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan}
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    rmse = float(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)))
    mae = float(mean_absolute_error(y_true_valid, y_pred_valid))
    if len(y_true_valid) < 2:
        corr = np.nan
    else:
        corr = float(np.corrcoef(y_true_valid, y_pred_valid)[0, 1])
    return {"rmse": rmse, "mae": mae, "corr": corr}


def plot_window(
    output_dir: Path,
    window_label: str,
    actual: pd.Series,
    baseline: pd.Series,
    weather: Optional[pd.Series],
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(actual.index, actual.values, label="实际功率", linewidth=1.5)
    plt.plot(baseline.index, baseline.values, label="历史基线", linewidth=1.2)
    if weather is not None:
        plt.plot(weather.index, weather.values, label="天气模型", linewidth=1.2)
    plt.xlabel("时间")
    plt.ylabel("功率")
    plt.title(f"测试周对比：{window_label}")
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"forecast_{window_label}.png", dpi=150)
    plt.close()


def plot_scatter(
    output_dir: Path,
    label: str,
    actual: pd.Series,
    predicted: pd.Series,
) -> None:
    valid = ~(actual.isna() | predicted.isna())
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(actual[valid], predicted[valid], s=10, alpha=0.6)
    plt.xlabel("实际功率")
    plt.ylabel("预测功率")
    plt.title(f"散点图：{label}")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"scatter_{label}.png", dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem 1: 光伏电站发电功率日前预测")
    parser.add_argument("--data", required=True, type=Path, help="CSV数据路径")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--baseline-days", type=int, default=7, help="基线模型取前几天平均")
    args = parser.parse_args()

    df, column_map = load_and_prepare(args.data)
    test_windows = build_test_windows(df.index)
    if not test_windows:
        raise ValueError("未找到测试周（每年2/5/8/11月最后一周）。请检查数据年份范围。")

    test_mask = pd.Series(False, index=df.index)
    for start, end in test_windows:
        test_mask |= (df.index >= start) & (df.index <= end)
    train_df = df.loc[~test_mask]

    daylight = daylight_mask(df, column_map)
    fallback_by_time = time_of_day_mean(train_df[column_map.power])

    baseline_predictions = []
    weather_predictions = []
    metrics_rows = []

    weather_available = len(column_map.weather_columns) > 0
    if weather_available:
        train_features = build_weather_features(train_df, column_map)
        train_target = train_df[column_map.power]
        train_daylight = daylight.loc[train_df.index]
        model = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        )
        model.fit(train_features.loc[train_daylight], train_target.loc[train_daylight])
    else:
        model = None

    plots_dir = args.output_dir / "plots"
    for start, end in test_windows:
        window_label = f"{start:%Y%m%d}_{end:%Y%m%d}"
        window_index = df.loc[start:end].index
        actual = df.loc[start:end, column_map.power]
        baseline = baseline_forecast(
            df,
            column_map,
            window_index,
            args.baseline_days,
            fallback_by_time,
        )
        baseline_predictions.append(baseline.rename("baseline"))

        if model is not None:
            window_features = build_weather_features(df.loc[start:end], column_map)
            weather_pred = pd.Series(model.predict(window_features), index=window_index)
            weather_predictions.append(weather_pred.rename("weather"))
        else:
            weather_pred = None

        eval_mask = daylight.loc[window_index]
        baseline_metrics = evaluate_metrics(actual[eval_mask], baseline[eval_mask])
        baseline_metrics.update({"window": window_label, "model": "baseline"})
        metrics_rows.append(baseline_metrics)

        if weather_pred is not None:
            weather_metrics = evaluate_metrics(actual[eval_mask], weather_pred[eval_mask])
            weather_metrics.update({"window": window_label, "model": "weather"})
            metrics_rows.append(weather_metrics)

        plot_window(plots_dir, window_label, actual, baseline, weather_pred)

    all_baseline = pd.concat(baseline_predictions).sort_index()
    all_actual = df.loc[all_baseline.index, column_map.power]
    plot_scatter(plots_dir, "baseline", all_actual, all_baseline)

    all_weather = None
    if weather_predictions:
        all_weather = pd.concat(weather_predictions).sort_index()
        plot_scatter(plots_dir, "weather", all_actual.loc[all_weather.index], all_weather)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "actual": all_actual,
            "baseline": all_baseline,
        }
    )
    if all_weather is not None:
        predictions_df["weather"] = all_weather
    predictions_df.to_csv(args.output_dir / "predictions.csv")

    print("完成：输出目录", args.output_dir.resolve())
    if not weather_available:
        print("注意：未检测到天气字段，只生成了历史基线模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

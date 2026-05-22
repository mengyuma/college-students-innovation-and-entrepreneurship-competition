#!/usr/bin/env python3
"""
Problem B: 嵌入式社区养老服务站的建设与优化问题

使用方法（示例）:
  python problem2_solution.py --base-dir . --output-dir outputs

默认读取附件:
  - 附件1：小区基础数据.xlsx
  - 附件2：服务需求数据.xlsx
  - 附件3：服务站建设与运营成本.xlsx
  - 附件4：小区间距离矩阵.xlsx
  - 附件5：满意度评分规则.xlsx
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp


COMMUNITY_FILE = "附件1：小区基础数据.xlsx"
DEMAND_FILE = "附件2：服务需求数据.xlsx"
STATION_COST_FILE = "附件3：服务站建设与运营成本.xlsx"
DISTANCE_FILE = "附件4：小区间距离矩阵.xlsx"
SATISFACTION_FILE = "附件5：满意度评分规则.xlsx"


@dataclass(frozen=True)
class ModelWeights:
    cost_weight: float
    distance_weight: float
    satisfaction_weight: float


@dataclass(frozen=True)
class ModelSettings:
    radius: float
    max_stations: Optional[int]
    budget_10k: Optional[float]
    time_limit: Optional[int]
    gap: Optional[float]
    price_multiplier: float
    weights: ModelWeights


def _clean_numeric(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value)
    cleaned = (
        text.replace("（", "(")
        .replace("）", ")")
        .replace(",", "")
        .replace("元", "")
        .replace(" ", "")
    )
    cleaned = cleaned.split("(")[0]
    cleaned = cleaned.replace("≤", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_percentage(value: object) -> float:
    numeric = _clean_numeric(value)
    if numeric > 1:
        return numeric / 100
    return numeric


def load_population_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="人口与老人结构", header=1)
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.dropna(subset=["小区编号"])
    numeric_cols = ["总人口", "60+老人数", "自理老人", "半失能老人", "失能老人", "人均月收入(元)"]
    for col in numeric_cols:
        df[col] = df[col].apply(_clean_numeric)
    return df.set_index("小区编号")


def load_service_demand(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="每位老人月均服务需求次数", header=1)
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.dropna(subset=["服务项目"])
    for col in ["自理", "半自理", "失能"]:
        df[col] = df[col].apply(_clean_numeric)
    return df.set_index("服务项目")


def load_service_prices(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="服务营收及支出", header=1)
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.dropna(subset=["服务项目"])
    df["单次服务营收（元）"] = df["单次服务营收（元）"].apply(_clean_numeric)
    df["单次服务直接支出（元）（基准价格）"] = df["单次服务直接支出（元）（基准价格）"].apply(_clean_numeric)
    return df.set_index("服务项目")


def load_spending_caps(path: Path) -> Dict[str, float]:
    df = pd.read_excel(path, sheet_name="月服务消费上限", header=0)
    df = df.dropna(subset=["老人类型"])
    caps = {}
    for _, row in df.iterrows():
        caps[str(row["老人类型"]).strip()] = _parse_percentage(
            row["月服务消费上限（占月收入比例）"]
        )
    return caps


def load_station_costs(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="服务站建设与运营成本", header=1)
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.dropna(subset=["站点规模"])
    df["一次性建设成本（万元）"] = df["一次性建设成本（万元）"].apply(_clean_numeric)
    df["日均固定管理成本（元/日）"] = df["日均固定管理成本（元/日）"].apply(_clean_numeric)
    df["日最大服务人次"] = df["日最大服务人次"].apply(_clean_numeric)
    df["年固定成本（元/年）"] = (
        df["一次性建设成本（万元）"] * 10000 / 20 + df["日均固定管理成本（元/日）"] * 365
    )
    return df.set_index("站点规模")


def load_distance_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="小区间距离矩阵", header=1)
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.dropna(subset=["组别"])
    df = df.set_index("组别")
    df = df.apply(lambda col: col.map(_clean_numeric))
    return df


def distance_satisfaction(distance: float) -> float:
    if distance <= 300:
        return 1.0
    if distance <= 500:
        return 0.9
    if distance <= 650:
        return 0.75
    if distance <= 1000:
        return 0.6
    return 0.0


def utilization_satisfaction(utilization: float) -> float:
    if utilization <= 0.60:
        return 1.0
    if utilization <= 0.75:
        return 0.93
    if utilization <= 0.85:
        return 0.85
    if utilization <= 0.95:
        return 0.72
    if utilization <= 1.00:
        return 0.60
    return 0.60


def price_satisfaction(multiplier: float) -> float:
    if multiplier <= 1.0:
        return 1.0
    if multiplier <= 1.1:
        return 0.9
    if multiplier <= 1.2:
        return 0.75
    return 0.6


def build_demand(
    population: pd.DataFrame,
    demand_per_elder: pd.DataFrame,
    price_table: pd.DataFrame,
    caps: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    elder_mapping = {
        "自理": "自理老人",
        "半自理": "半失能老人",
        "失能": "失能老人",
    }
    service_items = demand_per_elder.index.tolist()
    base_price = price_table["单次服务直接支出（元）（基准价格）"].reindex(service_items).fillna(0.0)
    monthly_records = []
    detail_records = []
    for community, row in population.iterrows():
        income = float(row["人均月收入(元)"])
        community_totals = {item: 0.0 for item in service_items}
        community_cost = 0.0
        for elder_type, pop_col in elder_mapping.items():
            elder_count = float(row[pop_col])
            if elder_count <= 0:
                continue
            demand_counts = demand_per_elder[elder_type] * elder_count
            total_cost = float((demand_counts * base_price).sum())
            cap_ratio = caps.get(f"{elder_type}老人", caps.get(elder_type, 0.0))
            cap_total = income * cap_ratio * elder_count
            if total_cost > cap_total and total_cost > 0:
                scale = cap_total / total_cost
                demand_counts = (demand_counts * scale).apply(np.floor)
                total_cost = float((demand_counts * base_price).sum())
            for item in service_items:
                count = float(demand_counts[item])
                community_totals[item] += count
                detail_records.append(
                    {
                        "小区编号": community,
                        "老人类型": elder_type,
                        "服务项目": item,
                        "月服务次数": count,
                    }
                )
            community_cost += total_cost
        monthly_services = sum(community_totals.values())
        monthly_records.append(
            {
                "小区编号": community,
                "月服务总次数": monthly_services,
                "日均服务总次数": monthly_services / 30,
                "月服务直接成本（元）": community_cost,
            }
            | {f"{item}_月服务次数": community_totals[item] for item in service_items}
        )
    summary_df = pd.DataFrame(monthly_records).set_index("小区编号")
    detail_df = pd.DataFrame(detail_records)
    return summary_df, detail_df


def solve_location_allocation(
    communities: List[str],
    demand_daily: Dict[str, float],
    distance: pd.DataFrame,
    station_costs: pd.DataFrame,
    settings: ModelSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sizes = station_costs.index.tolist()
    allowed_pairs = [
        (i, j)
        for i in communities
        for j in communities
        if distance.loc[i, j] <= settings.radius
    ]
    if not allowed_pairs:
        raise ValueError("未找到任何满足半径约束的可达组合。请放宽服务半径。")

    problem = pulp.LpProblem("facility_location", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("open", (communities, sizes), 0, 1, cat="Binary")
    y = pulp.LpVariable.dicts("assign", allowed_pairs, 0, 1, cat="Binary")

    for j in communities:
        problem += pulp.lpSum(x[j][s] for s in sizes) <= 1, f"one_size_{j}"

    for i in communities:
        problem += (
            pulp.lpSum(y[i, j] for i2, j in allowed_pairs if i2 == i) == 1,
            f"assign_{i}",
        )

    for i, j in allowed_pairs:
        problem += y[i, j] <= pulp.lpSum(x[j][s] for s in sizes), f"open_link_{i}_{j}"

    for j in communities:
        load = pulp.lpSum(demand_daily[i] * y[i, j] for i, j2 in allowed_pairs if j2 == j)
        capacity = pulp.lpSum(station_costs.loc[s, "日最大服务人次"] * x[j][s] for s in sizes)
        problem += load <= capacity, f"capacity_{j}"

    if settings.max_stations is not None:
        problem += (
            pulp.lpSum(x[j][s] for j in communities for s in sizes)
            <= settings.max_stations,
            "max_stations",
        )

    if settings.budget_10k is not None:
        problem += (
            pulp.lpSum(x[j][s] * station_costs.loc[s, "一次性建设成本（万元）"] for j in communities for s in sizes)
            <= settings.budget_10k,
            "budget",
        )

    fixed_cost = pulp.lpSum(
        x[j][s] * station_costs.loc[s, "年固定成本（元/年）"] for j in communities for s in sizes
    )
    distance_cost = pulp.lpSum(
        y[i, j] * demand_daily[i] * 365 * distance.loc[i, j] for i, j in allowed_pairs
    )
    satisfaction_proxy = pulp.lpSum(
        y[i, j] * demand_daily[i] * distance_satisfaction(distance.loc[i, j])
        for i, j in allowed_pairs
    )
    weights = settings.weights
    problem += (
        weights.cost_weight * fixed_cost
        + weights.distance_weight * distance_cost
        - weights.satisfaction_weight * satisfaction_proxy
    )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=settings.time_limit, gapRel=settings.gap)
    status = problem.solve(solver)
    if pulp.LpStatus[status] not in {"Optimal", "Feasible"}:
        raise RuntimeError(f"求解失败: {pulp.LpStatus[status]}")

    stations = []
    for j in communities:
        for s in sizes:
            if x[j][s].value() >= 0.5:
                stations.append(
                    {
                        "站点": j,
                        "规模": s,
                        "日最大服务人次": station_costs.loc[s, "日最大服务人次"],
                        "年固定成本（元/年）": station_costs.loc[s, "年固定成本（元/年）"],
                        "一次性建设成本（万元）": station_costs.loc[s, "一次性建设成本（万元）"],
                    }
                )
    station_df = pd.DataFrame(stations)

    assignments = []
    for i, j in allowed_pairs:
        if y[i, j].value() >= 0.5:
            assignments.append(
                {
                    "小区编号": i,
                    "服务站": j,
                    "距离（米）": distance.loc[i, j],
                    "日均服务次数": demand_daily[i],
                }
            )
    assign_df = pd.DataFrame(assignments)
    return station_df, assign_df


def evaluate_plan(
    station_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    station_costs: pd.DataFrame,
    price_multiplier: float,
    monthly_cost: pd.Series,
    distance: pd.DataFrame,
) -> Dict[str, float]:
    if station_df.empty or assign_df.empty:
        raise ValueError("方案为空，无法评估。")

    station_load = assign_df.groupby("服务站")["日均服务次数"].sum()
    station_capacity = station_df.set_index("站点")["日最大服务人次"]
    utilization = (station_load / station_capacity).fillna(0.0)
    station_s2 = utilization.apply(utilization_satisfaction)
    assign_df = assign_df.copy()
    assign_df["S1"] = assign_df["距离（米）"].apply(distance_satisfaction)
    assign_df["S2"] = assign_df["服务站"].map(station_s2)
    s3 = price_satisfaction(price_multiplier)
    assign_df["S3"] = s3
    assign_df["满意度"] = 0.2 * assign_df["S1"] + 0.3 * assign_df["S2"] + 0.5 * assign_df["S3"]

    total_demand = assign_df["日均服务次数"].sum()
    weighted_satisfaction = (assign_df["满意度"] * assign_df["日均服务次数"]).sum() / total_demand
    total_distance = (assign_df["距离（米）"] * assign_df["日均服务次数"]).sum()
    annual_fixed_cost = station_df["年固定成本（元/年）"].sum()
    annual_service_cost = monthly_cost.sum() * 12
    total_cost = annual_fixed_cost + annual_service_cost

    return {
        "年固定成本（元/年）": annual_fixed_cost,
        "年服务直接成本（元/年）": annual_service_cost,
        "年总成本（元/年）": total_cost,
        "日均加权出行距离（米）": total_distance / total_demand,
        "平均满意度": weighted_satisfaction,
    }


def build_baseline(
    communities: List[str],
    demand_daily: Dict[str, float],
    station_costs: pd.DataFrame,
    distance: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stations = []
    assignments = []
    sizes = station_costs.index.tolist()
    for community in communities:
        demand = demand_daily[community]
        eligible_sizes = [
            s for s in sizes if station_costs.loc[s, "日最大服务人次"] >= demand
        ]
        if eligible_sizes:
            size = min(eligible_sizes, key=lambda s: station_costs.loc[s, "日最大服务人次"])
        else:
            size = max(sizes, key=lambda s: station_costs.loc[s, "日最大服务人次"])
        stations.append(
            {
                "站点": community,
                "规模": size,
                "日最大服务人次": station_costs.loc[size, "日最大服务人次"],
                "年固定成本（元/年）": station_costs.loc[size, "年固定成本（元/年）"],
                "一次性建设成本（万元）": station_costs.loc[size, "一次性建设成本（万元）"],
            }
        )
        assignments.append(
            {
                "小区编号": community,
                "服务站": community,
                "距离（米）": distance.loc[community, community],
                "日均服务次数": demand,
            }
        )
    return pd.DataFrame(stations), pd.DataFrame(assignments)


def solve_and_report(
    summary_df: pd.DataFrame,
    station_costs: pd.DataFrame,
    distance: pd.DataFrame,
    settings: ModelSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    communities = summary_df.index.tolist()
    demand_daily = summary_df["日均服务总次数"].to_dict()
    station_df, assign_df = solve_location_allocation(
        communities, demand_daily, distance, station_costs, settings
    )
    metrics = evaluate_plan(
        station_df,
        assign_df,
        station_costs,
        settings.price_multiplier,
        summary_df["月服务直接成本（元）"],
        distance,
    )
    return station_df, assign_df, metrics


def parse_float_list(values: Optional[str]) -> List[float]:
    if not values:
        return []
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem B: 嵌入式社区养老服务站建设与优化")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="附件所在目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--radius", type=float, default=1000, help="服务半径（米）")
    parser.add_argument("--max-stations", type=int, default=None, help="最大站点数量")
    parser.add_argument("--budget-10k", type=float, default=None, help="建设预算上限（万元）")
    parser.add_argument("--time-limit", type=int, default=30, help="求解时间上限（秒）")
    parser.add_argument("--gap", type=float, default=0.01, help="可接受的相对最优差距")
    parser.add_argument("--price-multiplier", type=float, default=1.0, help="价格相对基准价倍率")
    parser.add_argument("--cost-weight", type=float, default=1.0, help="成本权重")
    parser.add_argument("--distance-weight", type=float, default=0.0005, help="距离权重")
    parser.add_argument("--satisfaction-weight", type=float, default=1.0, help="满意度代理权重")
    parser.add_argument("--sensitivity-budgets", type=str, default=None, help="预算敏感性（万元，逗号分隔）")
    parser.add_argument("--sensitivity-radius", type=str, default=None, help="半径敏感性（米，逗号分隔）")
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    population = load_population_data(base_dir / COMMUNITY_FILE)
    demand_per_elder = load_service_demand(base_dir / DEMAND_FILE)
    price_table = load_service_prices(base_dir / DEMAND_FILE)
    caps = load_spending_caps(base_dir / DEMAND_FILE)
    station_costs = load_station_costs(base_dir / STATION_COST_FILE)
    distance = load_distance_matrix(base_dir / DISTANCE_FILE)

    summary_df, detail_df = build_demand(population, demand_per_elder, price_table, caps)
    summary_df.to_csv(output_dir / "community_demand_summary.csv")
    detail_df.to_csv(output_dir / "community_demand_detail.csv", index=False)

    settings = ModelSettings(
        radius=args.radius,
        max_stations=args.max_stations,
        budget_10k=args.budget_10k,
        time_limit=args.time_limit,
        gap=args.gap,
        price_multiplier=args.price_multiplier,
        weights=ModelWeights(
            cost_weight=args.cost_weight,
            distance_weight=args.distance_weight,
            satisfaction_weight=args.satisfaction_weight,
        ),
    )

    station_df, assign_df, metrics = solve_and_report(
        summary_df, station_costs, distance, settings
    )
    station_df.to_csv(output_dir / "station_plan.csv", index=False)
    assign_df.to_csv(output_dir / "assignment_plan.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "solution_metrics.csv", index=False)

    baseline_station, baseline_assign = build_baseline(
        summary_df.index.tolist(), summary_df["日均服务总次数"].to_dict(), station_costs, distance
    )
    baseline_metrics = evaluate_plan(
        baseline_station,
        baseline_assign,
        station_costs,
        settings.price_multiplier,
        summary_df["月服务直接成本（元）"],
        distance,
    )
    baseline_station.to_csv(output_dir / "baseline_station_plan.csv", index=False)
    baseline_assign.to_csv(output_dir / "baseline_assignment_plan.csv", index=False)
    pd.DataFrame([baseline_metrics]).to_csv(output_dir / "baseline_metrics.csv", index=False)

    sensitivity_records = []
    for budget in parse_float_list(args.sensitivity_budgets):
        scenario_settings = ModelSettings(
            **{**settings.__dict__, "budget_10k": budget},
        )
        _, _, scenario_metrics = solve_and_report(summary_df, station_costs, distance, scenario_settings)
        scenario_metrics["情景"] = f"预算={budget}万元"
        sensitivity_records.append(scenario_metrics)

    for radius in parse_float_list(args.sensitivity_radius):
        scenario_settings = ModelSettings(
            **{**settings.__dict__, "radius": radius},
        )
        _, _, scenario_metrics = solve_and_report(summary_df, station_costs, distance, scenario_settings)
        scenario_metrics["情景"] = f"半径={radius}m"
        sensitivity_records.append(scenario_metrics)

    if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(
            output_dir / "sensitivity_metrics.csv", index=False
        )

    print("完成：输出目录", output_dir.resolve())
    print("最优方案关键指标:", metrics)
    print("基准方案关键指标:", baseline_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

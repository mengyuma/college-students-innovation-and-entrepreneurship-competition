#!/usr/bin/env python3
"""
Problem B: 嵌入式社区养老服务站建设与优化.

示例:
  python problem2_solution.py \
    --data-dir /path/to/attachments \
    --output-dir outputs \
    --station-count 3 \
    --station-size auto \
    --price-multiplier 1.0
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


POPULATION_FILE = "附件1：小区基础数据.xlsx"
SERVICE_FILE = "附件2：服务需求数据.xlsx"
STATION_COST_FILE = "附件3：服务站建设与运营成本.xlsx"
DISTANCE_FILE = "附件4：小区间距离矩阵.xlsx"


distance_bins = [(300, 1.0), (500, 0.9), (650, 0.75), (1000, 0.6)]
utilization_bins = [(0.60, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.00, 0.6)]
price_bins = [(1.00, 1.0), (1.10, 0.9), (1.20, 0.75)]


def _normalize_text(value: object) -> str:
    return str(value).replace(" ", "").replace("\n", "").replace("\r", "")


def _first_match(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {_normalize_text(col): col for col in columns}
    for candidate in candidates:
        candidate_key = _normalize_text(candidate)
        if candidate_key in normalized:
            return normalized[candidate_key]
    return None


def _needs_header_row(df: pd.DataFrame, sheet_name: str) -> bool:
    unnamed_count = sum(str(col).startswith("Unnamed") for col in df.columns)
    if unnamed_count >= max(1, len(df.columns) // 2):
        return True
    first_col = _normalize_text(df.columns[0])
    return _normalize_text(sheet_name) in first_col


def _load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=0)
    if raw.empty:
        raise ValueError(f"{path.name}:{sheet_name} 空表")
    if _needs_header_row(raw, sheet_name):
        header = raw.iloc[0].tolist()
        df = raw.iloc[1:].copy()
        df.columns = [
            _normalize_text(col) if not pd.isna(col) else f"col_{idx}"
            for idx, col in enumerate(header)
        ]
    else:
        df = raw.copy()
        df.columns = [_normalize_text(col) for col in df.columns]
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def _parse_percentage(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        ratio = float(value)
        return ratio / 100 if ratio > 1 else ratio
    text = str(value)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    ratio = float(match.group(1))
    return ratio / 100 if ratio > 1 else ratio


def _score_from_bins(value: float, bins: List[Tuple[float, float]], default: float) -> float:
    for upper, score in bins:
        if value <= upper:
            return score
    return default


def _score_price(multiplier: float) -> float:
    for upper, score in price_bins:
        if multiplier <= upper:
            return score
    return 0.6


def _score_distance(distance: float) -> float:
    return _score_from_bins(distance, distance_bins, 0.6)


def _score_utilization(utilization: float) -> float:
    return _score_from_bins(utilization, utilization_bins, 0.6)


@dataclass(frozen=True)
class StationType:
    name: str
    construction_cost: float
    daily_fixed_cost: float
    daily_capacity: float


@dataclass
class DemandProfile:
    demand_by_service: pd.DataFrame
    monthly_total: pd.Series
    daily_total: pd.Series


@dataclass
class SolutionResult:
    stations: pd.DataFrame
    assignments: pd.DataFrame
    summary: Dict[str, float]


def load_population_data(path: Path) -> pd.DataFrame:
    df = _load_sheet(path, "人口与老人结构")
    columns = {
        "community": ["小区编号", "社区编号", "编号"],
        "total_population": ["总人口"],
        "elder_total": ["60+老人数", "60+老人", "60岁以上"],
        "self_care": ["自理老人", "自理"],
        "semi_disabled": ["半失能老人", "半自理老人", "半自理", "半失能"],
        "disabled": ["失能老人", "失能"],
        "income": ["人均月收入(元)", "人均月收入", "月收入"],
    }
    rename_map = {}
    for key, candidates in columns.items():
        match = _first_match(df.columns, candidates)
        if match is None:
            raise ValueError(f"人口数据缺少字段: {key}")
        rename_map[match] = key
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["community"])
    numeric_cols = [
        "total_population",
        "elder_total",
        "self_care",
        "semi_disabled",
        "disabled",
        "income",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["community"] = df["community"].astype(str).str.strip()
    return df


def load_service_demand(path: Path) -> pd.DataFrame:
    df = _load_sheet(path, "每位老人月均服务需求次数")
    columns = {
        "service": ["服务项目"],
        "self_care": ["自理"],
        "semi_disabled": ["半自理", "半失能"],
        "disabled": ["失能"],
    }
    rename_map = {}
    for key, candidates in columns.items():
        match = _first_match(df.columns, candidates)
        if match is None:
            raise ValueError(f"服务需求数据缺少字段: {key}")
        rename_map[match] = key
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["service"])
    df["service"] = df["service"].astype(str).str.strip()
    for col in ["self_care", "semi_disabled", "disabled"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_service_finance(path: Path) -> pd.DataFrame:
    df = _load_sheet(path, "服务营收及支出")
    columns = {
        "service": ["服务项目"],
        "price": ["单次服务营收（元）", "单次服务营收"],
        "direct_cost": ["单次服务直接支出（元）（基准价格）", "单次服务直接支出"],
    }
    rename_map = {}
    for key, candidates in columns.items():
        match = _first_match(df.columns, candidates)
        if match is None:
            raise ValueError(f"营收及支出数据缺少字段: {key}")
        rename_map[match] = key
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["service"])
    df["service"] = df["service"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["direct_cost"] = pd.to_numeric(df["direct_cost"], errors="coerce")
    return df


def load_spending_limits(path: Path) -> Dict[str, float]:
    df = _load_sheet(path, "月服务消费上限")
    columns = {
        "elder_type": ["老人类型"],
        "limit": ["月服务消费上限（占月收入比例）", "月服务消费上限"],
    }
    rename_map = {}
    for key, candidates in columns.items():
        match = _first_match(df.columns, candidates)
        if match is None:
            raise ValueError(f"消费上限数据缺少字段: {key}")
        rename_map[match] = key
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["elder_type"])
    limits: Dict[str, float] = {}
    for _, row in df.iterrows():
        limit = _parse_percentage(row["limit"])
        if limit is None:
            continue
        elder_type = str(row["elder_type"]).strip()
        limits[elder_type] = float(limit)
    return limits


def load_station_costs(path: Path) -> Dict[str, StationType]:
    df = _load_sheet(path, "服务站建设与运营成本")
    columns = {
        "size": ["站点规模"],
        "construction": ["一次性建设成本（万元）", "一次性建设成本"],
        "daily_fixed": ["日均固定管理成本（元/日）", "日均固定管理成本"],
        "capacity": ["日最大服务人次", "日最大服务"],
    }
    rename_map = {}
    for key, candidates in columns.items():
        match = _first_match(df.columns, candidates)
        if match is None:
            raise ValueError(f"服务站成本数据缺少字段: {key}")
        rename_map[match] = key
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["size"])
    station_types: Dict[str, StationType] = {}
    for _, row in df.iterrows():
        size = str(row["size"]).strip()
        construction = pd.to_numeric(row["construction"], errors="coerce")
        daily_fixed = pd.to_numeric(row["daily_fixed"], errors="coerce")
        capacity = pd.to_numeric(row["capacity"], errors="coerce")
        if pd.isna(construction) or pd.isna(daily_fixed) or pd.isna(capacity):
            continue
        station_types[size] = StationType(
            name=size,
            construction_cost=float(construction) * 10000,
            daily_fixed_cost=float(daily_fixed),
            daily_capacity=float(capacity),
        )
    if not station_types:
        raise ValueError("未加载到服务站成本数据")
    return station_types


def load_distance_matrix(path: Path) -> pd.DataFrame:
    df = _load_sheet(path, "小区间距离矩阵")
    first_column = df.columns[0]
    df = df.rename(columns={first_column: "community"})
    df = df.dropna(subset=["community"])
    df["community"] = df["community"].astype(str).str.strip()
    df = df.set_index("community")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def compute_demand_profile(
    population: pd.DataFrame,
    demand_rates: pd.DataFrame,
    finance: pd.DataFrame,
    spending_limits: Dict[str, float],
    price_multiplier: float,
) -> DemandProfile:
    services = demand_rates["service"].tolist()
    price_map = finance.set_index("service")["price"].to_dict()
    direct_cost_map = finance.set_index("service")["direct_cost"].to_dict()

    demand_records = []
    for _, row in population.iterrows():
        community = row["community"]
        income = float(row["income"])
        counts = {
            "self_care": float(row["self_care"]),
            "semi_disabled": float(row["semi_disabled"]),
            "disabled": float(row["disabled"]),
        }
        elder_type_names = {
            "self_care": ["自理老人", "自理"],
            "semi_disabled": ["半失能老人", "半自理老人", "半失能", "半自理"],
            "disabled": ["失能老人", "失能"],
        }
        service_totals = {service: 0.0 for service in services}
        for elder_key, count in counts.items():
            if count <= 0:
                continue
            limit_ratio = None
            for name in elder_type_names[elder_key]:
                if name in spending_limits:
                    limit_ratio = spending_limits[name]
                    break
            if limit_ratio is None:
                limit_ratio = 0.25
            base_cost = 0.0
            for _, rate_row in demand_rates.iterrows():
                service = rate_row["service"]
                frequency = float(rate_row[elder_key])
                price = float(price_map.get(service, 0.0)) * price_multiplier
                base_cost += frequency * price
            budget = income * limit_ratio
            scale = 1.0
            if base_cost > 0 and budget > 0:
                scale = min(1.0, budget / base_cost)
            for _, rate_row in demand_rates.iterrows():
                service = rate_row["service"]
                frequency = float(rate_row[elder_key]) * scale
                service_totals[service] += frequency * count
        record = {"community": community}
        record.update(service_totals)
        demand_records.append(record)

    demand_df = pd.DataFrame(demand_records).set_index("community")
    demand_df = demand_df.fillna(0)
    demand_df["total_monthly"] = demand_df.sum(axis=1)
    monthly_total = demand_df["total_monthly"]
    daily_total = monthly_total / 30

    for service in services:
        if service not in direct_cost_map:
            direct_cost_map[service] = 0.0
    demand_df.attrs["price_map"] = price_map
    demand_df.attrs["direct_cost_map"] = direct_cost_map
    return DemandProfile(demand_by_service=demand_df, monthly_total=monthly_total, daily_total=daily_total)


def assign_communities(
    communities: List[str],
    stations: Tuple[str, ...],
    distance_matrix: pd.DataFrame,
    max_distance: float,
) -> Dict[str, Optional[str]]:
    assignments: Dict[str, Optional[str]] = {}
    for community in communities:
        distances = distance_matrix.loc[community, list(stations)]
        nearest_station = distances.idxmin()
        nearest_distance = float(distances.loc[nearest_station])
        if nearest_distance > max_distance:
            assignments[community] = None
        else:
            assignments[community] = str(nearest_station)
    return assignments


def choose_station_type(
    station_types: Dict[str, StationType],
    daily_load: float,
    preferred: str,
) -> StationType:
    if preferred != "auto":
        if preferred not in station_types:
            raise ValueError(f"未知站点规模: {preferred}")
        return station_types[preferred]
    sorted_types = sorted(station_types.values(), key=lambda item: item.daily_capacity)
    for station_type in sorted_types:
        if daily_load <= station_type.daily_capacity:
            return station_type
    return sorted_types[-1]


def evaluate_solution(
    communities: List[str],
    stations: Tuple[str, ...],
    distance_matrix: pd.DataFrame,
    demand: DemandProfile,
    station_types: Dict[str, StationType],
    price_multiplier: float,
    max_distance: float,
    preferred_size: str,
    satisfaction_weight: float,
    unmet_penalty: float,
    overflow_penalty: float,
) -> SolutionResult:
    assignments = assign_communities(communities, stations, distance_matrix, max_distance)
    assignments_df = []
    station_loads = {station: 0.0 for station in stations}
    unserved_demand = 0.0

    for community in communities:
        monthly_demand = float(demand.monthly_total.loc[community])
        daily_demand = float(demand.daily_total.loc[community])
        station = assignments[community]
        if station is None:
            unserved_demand += monthly_demand
        else:
            station_loads[station] += daily_demand

    station_rows = []
    station_utilization = {}
    overflow_demand = 0.0
    for station, daily_load in station_loads.items():
        station_type = choose_station_type(station_types, daily_load, preferred_size)
        utilization = daily_load / station_type.daily_capacity if station_type.daily_capacity > 0 else 0
        if utilization > 1.0:
            overflow_demand += (utilization - 1.0) * station_type.daily_capacity * 30
        station_utilization[station] = utilization
        station_rows.append(
            {
                "station": station,
                "size": station_type.name,
                "daily_capacity": station_type.daily_capacity,
                "daily_load": daily_load,
                "utilization": utilization,
                "s2": _score_utilization(min(utilization, 1.0)),
                "construction_cost": station_type.construction_cost,
                "monthly_fixed_cost": station_type.daily_fixed_cost * 30,
            }
        )

    station_df = pd.DataFrame(station_rows).set_index("station")

    s3 = _score_price(price_multiplier)
    total_weighted_satisfaction = 0.0
    total_demand = float(demand.monthly_total.sum())
    for community in communities:
        monthly_demand = float(demand.monthly_total.loc[community])
        station = assignments[community]
        if station is None:
            s1 = 0.0
            s2 = 0.0
            satisfaction = 0.0
            distance = math.nan
        else:
            distance = float(distance_matrix.loc[community, station])
            s1 = _score_distance(distance)
            s2 = _score_utilization(min(station_utilization[station], 1.0))
            satisfaction = 0.2 * s1 + 0.3 * s2 + 0.5 * s3
        total_weighted_satisfaction += satisfaction * monthly_demand
        assignments_df.append(
            {
                "community": community,
                "station": station or "",
                "distance_m": distance,
                "monthly_demand": monthly_demand,
                "daily_demand": monthly_demand / 30,
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "satisfaction": satisfaction,
                "served": station is not None,
            }
        )

    assignments_df = pd.DataFrame(assignments_df).set_index("community")
    avg_satisfaction = total_weighted_satisfaction / total_demand if total_demand > 0 else 0.0

    price_map = demand.demand_by_service.attrs.get("price_map", {})
    direct_cost_map = demand.demand_by_service.attrs.get("direct_cost_map", {})
    variable_cost = 0.0
    revenue = 0.0
    for service, series in demand.demand_by_service.drop(columns=["total_monthly"]).items():
        service_total = float(series.sum())
        variable_cost += service_total * float(direct_cost_map.get(service, 0.0))
        revenue += service_total * float(price_map.get(service, 0.0)) * price_multiplier

    construction_cost = station_df["construction_cost"].sum() if not station_df.empty else 0.0
    monthly_fixed_cost = station_df["monthly_fixed_cost"].sum() if not station_df.empty else 0.0
    total_cost = construction_cost + monthly_fixed_cost + variable_cost

    objective_score = (
        total_cost
        + unmet_penalty * unserved_demand
        + overflow_penalty * overflow_demand
        - satisfaction_weight * avg_satisfaction * total_demand
    )

    summary = {
        "station_count": len(stations),
        "total_cost": total_cost,
        "construction_cost": construction_cost,
        "monthly_fixed_cost": monthly_fixed_cost,
        "variable_cost": variable_cost,
        "revenue": revenue,
        "total_demand": total_demand,
        "avg_satisfaction": avg_satisfaction,
        "unserved_demand": unserved_demand,
        "overflow_demand": overflow_demand,
        "price_multiplier": price_multiplier,
        "objective_score": objective_score,
    }
    return SolutionResult(stations=station_df, assignments=assignments_df, summary=summary)


def search_best_solution(
    communities: List[str],
    station_count: int,
    distance_matrix: pd.DataFrame,
    demand: DemandProfile,
    station_types: Dict[str, StationType],
    price_multiplier: float,
    max_distance: float,
    preferred_size: str,
    satisfaction_weight: float,
    unmet_penalty: float,
    overflow_penalty: float,
) -> SolutionResult:
    best_result: Optional[SolutionResult] = None
    for stations in itertools.combinations(communities, station_count):
        result = evaluate_solution(
            communities,
            stations,
            distance_matrix,
            demand,
            station_types,
            price_multiplier,
            max_distance,
            preferred_size,
            satisfaction_weight,
            unmet_penalty,
            overflow_penalty,
        )
        if best_result is None or result.summary["objective_score"] < best_result.summary["objective_score"]:
            best_result = result
    if best_result is None:
        raise ValueError("未找到可行的站点组合")
    return best_result


def build_tradeoff_curve(
    communities: List[str],
    max_stations: int,
    distance_matrix: pd.DataFrame,
    demand: DemandProfile,
    station_types: Dict[str, StationType],
    price_multiplier: float,
    max_distance: float,
    preferred_size: str,
    satisfaction_weight: float,
    unmet_penalty: float,
    overflow_penalty: float,
) -> Tuple[pd.DataFrame, SolutionResult]:
    curve_rows = []
    best_overall: Optional[SolutionResult] = None
    for k in range(1, max_stations + 1):
        result = search_best_solution(
            communities,
            k,
            distance_matrix,
            demand,
            station_types,
            price_multiplier,
            max_distance,
            preferred_size,
            satisfaction_weight,
            unmet_penalty,
            overflow_penalty,
        )
        row = dict(result.summary)
        row["station_count"] = k
        curve_rows.append(row)
        if best_overall is None or result.summary["objective_score"] < best_overall.summary["objective_score"]:
            best_overall = result
    if best_overall is None:
        raise ValueError("未找到可行的站点组合")
    curve_df = pd.DataFrame(curve_rows)
    return curve_df, best_overall


def save_outputs(output_dir: Path, result: SolutionResult, demand: DemandProfile) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.stations.to_csv(output_dir / "station_plan.csv")
    result.assignments.to_csv(output_dir / "community_assignment.csv")
    demand.demand_by_service.to_csv(output_dir / "service_demand.csv")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result.summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Problem B: 社区养老服务站选址与优化")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="包含附件数据的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_b"),
        help="输出目录",
    )
    parser.add_argument("--station-count", type=int, default=3, help="站点数量")
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="若提供，将生成1..max的权衡曲线",
    )
    parser.add_argument(
        "--station-size",
        type=str,
        default="auto",
        choices=["auto", "小型", "中型", "大型"],
        help="站点规模（auto为自动匹配容量）",
    )
    parser.add_argument("--price-multiplier", type=float, default=1.0, help="价格倍率")
    parser.add_argument("--max-distance", type=float, default=1000.0, help="最大服务距离(米)")
    parser.add_argument("--satisfaction-weight", type=float, default=500.0, help="满意度权重")
    parser.add_argument("--unmet-penalty", type=float, default=2000.0, help="未覆盖需求惩罚系数")
    parser.add_argument("--overflow-penalty", type=float, default=1000.0, help="超载需求惩罚系数")
    args = parser.parse_args()

    data_dir = args.data_dir
    population = load_population_data(data_dir / POPULATION_FILE)
    demand_rates = load_service_demand(data_dir / SERVICE_FILE)
    finance = load_service_finance(data_dir / SERVICE_FILE)
    spending_limits = load_spending_limits(data_dir / SERVICE_FILE)
    station_types = load_station_costs(data_dir / STATION_COST_FILE)
    distance_matrix = load_distance_matrix(data_dir / DISTANCE_FILE)

    communities = population["community"].tolist()
    missing = set(communities) - set(distance_matrix.index)
    if missing:
        raise ValueError(f"距离矩阵缺少小区: {sorted(missing)}")
    distance_matrix = distance_matrix.loc[communities, communities]

    demand_profile = compute_demand_profile(
        population,
        demand_rates,
        finance,
        spending_limits,
        args.price_multiplier,
    )

    if args.max_stations:
        tradeoff_df, best_result = build_tradeoff_curve(
            communities,
            args.max_stations,
            distance_matrix,
            demand_profile,
            station_types,
            args.price_multiplier,
            args.max_distance,
            args.station_size,
            args.satisfaction_weight,
            args.unmet_penalty,
            args.overflow_penalty,
        )
        tradeoff_df.to_csv(args.output_dir / "tradeoff_curve.csv", index=False)
    else:
        best_result = search_best_solution(
            communities,
            args.station_count,
            distance_matrix,
            demand_profile,
            station_types,
            args.price_multiplier,
            args.max_distance,
            args.station_size,
            args.satisfaction_weight,
            args.unmet_penalty,
            args.overflow_penalty,
        )

    save_outputs(args.output_dir, best_result, demand_profile)
    print("完成：输出目录", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

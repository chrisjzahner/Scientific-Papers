# Platelet wastage model, final corrected run (3 usable days vs 5).
#
# Wastage is expired units / units received.
# A shortage day is demand greater than inventory on hand.
# Service of 99/98/96% means shortage days held to 1/2/4%.
# Baseline is 3 usable days left at receipt; the shelf-life run is 5.
# Redistribution is source-side only. A saved unit leaves the sending
# hospital and counts as saved. The receiving hospital is not simulated.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

YEARS = 180
WARMUP_DAYS = 56
DEMAND_SEED = 20260922
SHIPMENT_SEED = 713

SCENARIOS = {
    "Very low (~1/day)": (1.0, math.sqrt((3.7**2) / 5.2)),
    "Lower (5.2/day)": (5.2, 3.7),
    "Medium (11.8/day)": (11.8, 4.4),
    "High (35.4/day)": (35.4, 9.2),
}

SHORTAGE_TARGETS = [1, 2, 4]  # percent of days with unmet demand


def negative_binomial_params(mean, sd):
    var = sd**2
    if var <= mean:
        return None
    r = mean**2 / (var - mean)
    p = r / (r + mean)
    return r, p


def generate_demand(mean, sd):
    rng = np.random.default_rng(DEMAND_SEED)
    total_days = WARMUP_DAYS + 365
    params = negative_binomial_params(mean, sd)
    if params is None:
        return rng.poisson(mean, (total_days, YEARS)).astype(np.int16)
    r, p = params
    return rng.negative_binomial(r, p, (total_days, YEARS)).astype(np.int16)


def shipment_days_per_week(n):
    raw = np.linspace(0, 7, n, endpoint=False)
    days = sorted(set(int(round(x)) % 7 for x in raw))
    for d in range(7):
        if len(days) >= n:
            break
        if d not in days:
            days.append(d)
    return set(sorted(days[:n]))


def simulate_fixed_policy(
    demand,
    shipments_per_week,
    usable_days_at_receipt,
    average_shipment_quantity,
    redistribution_efficiency=0.0,
):
    total_days, n_years = demand.shape
    inventory = np.zeros((n_years, usable_days_at_receipt), dtype=np.int32)
    ship_days = shipment_days_per_week(shipments_per_week)
    rng = np.random.default_rng(SHIPMENT_SEED)

    received = 0.0
    expired = 0.0
    shortage_days = 0.0
    unmet_units = 0.0
    total_demand = 0.0
    redistributed = 0.0

    q_floor = int(np.floor(average_shipment_quantity))
    q_fraction = average_shipment_quantity - q_floor

    for day in range(total_days):
        # Bucket 0 expires at the start of the day, before issue.
        expiring_now = inventory[:, 0].copy()
        inventory[:, :-1] = inventory[:, 1:]
        inventory[:, -1] = 0

        if day % 7 in ship_days:
            add = np.full(n_years, q_floor, dtype=np.int32)
            if q_fraction > 0:
                add += (rng.random(n_years) < q_fraction).astype(np.int32)
            inventory[:, -1] += add
            if day >= WARMUP_DAYS:
                received += add.sum()

        remaining = demand[day].astype(np.int32).copy()
        for age_bucket in range(usable_days_at_receipt):
            take = np.minimum(inventory[:, age_bucket], remaining)
            inventory[:, age_bucket] -= take
            remaining -= take

        if day >= WARMUP_DAYS and redistribution_efficiency > 0:
            at_risk = inventory[:, 0].copy()
            saved = np.floor(at_risk * redistribution_efficiency).astype(np.int32)
            residual = at_risk * redistribution_efficiency - saved
            saved += (rng.random(n_years) < residual).astype(np.int32)
            inventory[:, 0] -= saved
            redistributed += saved.sum()

        if day >= WARMUP_DAYS:
            expired += expiring_now.sum()
            shortage_days += (remaining > 0).sum()
            unmet_units += remaining.sum()
            total_demand += demand[day].sum()

    return {
        "wastage_pct": 100 * expired / received,
        "shortage_day_pct": 100 * shortage_days / (n_years * 365),
        "unmet_demand_pct": 100 * unmet_units / total_demand,
        "redistributed_pct_of_received": 100 * redistributed / received if received else 0.0,
    }


def find_minimum_wastage_policy(
    demand,
    mean_daily_demand,
    shipments_per_week,
    usable_days_at_receipt,
    shortage_target_pct,
    redistribution_efficiency=0.0,
):
    """Smallest mean shipment that keeps shortage days at or under the target.

    A fractional size is stochastic. 4.3 means 4 units, and a fifth unit on 30% of deliveries.
    """
    base = max(0.05, mean_daily_demand * 7 / shipments_per_week)
    low = max(0.02, base * 0.15)
    high = max(0.3, base * 3)

    for _ in range(10):
        result = simulate_fixed_policy(
            demand,
            shipments_per_week,
            usable_days_at_receipt,
            high,
            redistribution_efficiency,
        )
        if result["shortage_day_pct"] <= shortage_target_pct:
            break
        high *= 1.5

    for _ in range(10):
        mid = (low + high) / 2
        result = simulate_fixed_policy(
            demand,
            shipments_per_week,
            usable_days_at_receipt,
            mid,
            redistribution_efficiency,
        )
        if result["shortage_day_pct"] <= shortage_target_pct:
            high = mid
        else:
            low = mid

    result = simulate_fixed_policy(
        demand,
        shipments_per_week,
        usable_days_at_receipt,
        high,
        redistribution_efficiency,
    )
    result["average_shipment_quantity"] = high
    return result


DEMAND = {name: generate_demand(*params) for name, params in SCENARIOS.items()}
_cache = {}


def policy(
    scenario_name,
    shortage_target_pct,
    shipments_per_week=3,
    usable_days_at_receipt=3,
    redistribution_efficiency=0.0,
):
    key = (
        scenario_name,
        shortage_target_pct,
        shipments_per_week,
        usable_days_at_receipt,
        round(redistribution_efficiency, 6),
    )
    if key not in _cache:
        mean = SCENARIOS[scenario_name][0]
        _cache[key] = find_minimum_wastage_policy(
            DEMAND[scenario_name],
            mean,
            shipments_per_week,
            usable_days_at_receipt,
            shortage_target_pct,
            redistribution_efficiency,
        )
    return _cache[key]


def exact_break_even_redistribution(scenario_name, shortage_target_pct):
    """Redistribution efficiency that matches 5-day wastage at the same service level."""
    target_wastage = policy(
        scenario_name, shortage_target_pct, 3, 5, 0.0
    )["wastage_pct"]

    low, high = 0.0, 1.0

    if policy(
        scenario_name, shortage_target_pct, 3, 3, 1.0
    )["wastage_pct"] > target_wastage:
        return 1.0

    for _ in range(12):
        mid = (low + high) / 2
        wastage = policy(
            scenario_name, shortage_target_pct, 3, 3, mid
        )["wastage_pct"]
        if wastage <= target_wastage:
            high = mid
        else:
            low = mid

    return high


full_rows = []
figure1_rows = []
figure2_rows = []
figure3_rows = []
figure4_rows = []

for scenario_name in SCENARIOS:
    for target in SHORTAGE_TARGETS:
        baseline = policy(scenario_name, target, 3, 3, 0.0)
        shelf5 = policy(scenario_name, target, 3, 5, 0.0)
        ship5 = policy(scenario_name, target, 5, 3, 0.0)
        ship7 = policy(scenario_name, target, 7, 3, 0.0)
        break_even = exact_break_even_redistribution(scenario_name, target)
        red = policy(scenario_name, target, 3, 3, break_even)
        combo = policy(scenario_name, target, 3, 5, break_even)

        scenarios_to_save = [
            ("Baseline: 3 usable days at receipt", baseline, 0.0),
            ("5 usable days at receipt", shelf5, 0.0),
            ("5 shipments/week, 3 usable days", ship5, 0.0),
            ("7 shipments/week, 3 usable days", ship7, 0.0),
            (f"Break-even redistribution ({break_even*100:.2f}%)", red, break_even),
            (
                f"5 usable days + break-even redistribution ({break_even*100:.2f}%)",
                combo,
                break_even,
            ),
        ]

        for label, result, red_eff in scenarios_to_save:
            full_rows.append({
                "Hospital volume": scenario_name,
                "Allowed shortage days (%)": target,
                "Service level (%)": 100 - target,
                "Scenario": label,
                "Wastage (%)": result["wastage_pct"],
                "Achieved shortage days (%)": result["shortage_day_pct"],
                "Unmet demand (%)": result["unmet_demand_pct"],
                "Average shipment quantity": result["average_shipment_quantity"],
                "Redistribution efficiency (%)": red_eff * 100,
            })

        figure1_rows.append({
            "Hospital volume": scenario_name,
            "Service level (%)": 100 - target,
            "Modeled wastage (%)": baseline["wastage_pct"],
        })

        if target == 1:
            figure2_rows.append({
                "Hospital volume": scenario_name,
                "3 to 5 usable days (pp)": baseline["wastage_pct"] - shelf5["wastage_pct"],
                "3 to 5 shipments/week (pp)": baseline["wastage_pct"] - ship5["wastage_pct"],
                "3 to 7 shipments/week (pp)": baseline["wastage_pct"] - ship7["wastage_pct"],
            })
            figure3_rows.append({
                "Hospital volume": scenario_name,
                "Baseline 3-day wastage (%)": baseline["wastage_pct"],
                "5-day wastage (%)": shelf5["wastage_pct"],
                "Shelf-life benefit (pp)": baseline["wastage_pct"] - shelf5["wastage_pct"],
                "Break-even redistribution efficiency (%)": break_even * 100,
            })

        if scenario_name == "Very low (~1/day)":
            figure4_rows.append({
                "Service level (%)": 100 - target,
                "Baseline 3-day wastage (%)": baseline["wastage_pct"],
                "5-day wastage (%)": shelf5["wastage_pct"],
                "Break-even redistribution efficiency (%)": break_even * 100,
            })


# Figure 3 holds redistribution at the very-low-volume 99% break-even.
very_low = "Very low (~1/day)"
fixed_red = exact_break_even_redistribution(very_low, 1)

figure5_rows = []
for target in SHORTAGE_TARGETS:
    baseline = policy(very_low, target, 3, 3, 0.0)
    shelf5 = policy(very_low, target, 3, 5, 0.0)
    red = policy(very_low, target, 3, 3, fixed_red)
    combo = policy(very_low, target, 3, 5, fixed_red)

    figure5_rows.append({
        "Service level (%)": 100 - target,
        "3 to 5 usable days (pp)": baseline["wastage_pct"] - shelf5["wastage_pct"],
        f"{fixed_red*100:.1f}% redistribution (pp)": baseline["wastage_pct"] - red["wastage_pct"],
        f"3 to 5 usable days + {fixed_red*100:.1f}% redistribution (pp)":
            baseline["wastage_pct"] - combo["wastage_pct"],
        "Baseline 3-day wastage (%)": baseline["wastage_pct"],
        "5-day wastage (%)": shelf5["wastage_pct"],
        "Redistribution-only wastage (%)": red["wastage_pct"],
        "Combined wastage (%)": combo["wastage_pct"],
        "Fixed redistribution efficiency (%)": fixed_red * 100,
    })


pd.DataFrame(figure1_rows).to_csv(
    OUT / "Figure_1_data.csv", index=False
)
pd.DataFrame(figure2_rows).to_csv(
    OUT / "Figure_2_data.csv", index=False
)
pd.DataFrame(figure3_rows).to_csv(
    OUT / "Supplementary_Figure_1_data.csv", index=False
)
pd.DataFrame(figure4_rows).to_csv(
    OUT / "Supplementary_Figure_2_data.csv", index=False
)
pd.DataFrame(figure5_rows).to_csv(
    OUT / "Figure_3_data.csv", index=False
)
pd.DataFrame(full_rows).to_csv(
    OUT / "Full_model_results.csv", index=False
)

print(f"Very-low-volume 99% break-even redistribution: {fixed_red*100:.2f}%")
print("Results written to:", OUT.resolve())

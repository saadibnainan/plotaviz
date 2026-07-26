#!/usr/bin/env python3
"""Generate the sample datasets in ``samples/``.

Each one is synthetic, deliberately small, and shaped to exercise a *different* path through the
selection engine, so a new contributor can see all the major branches without hunting for data:

======================================  ======================================================
File                                     Path it exercises
======================================  ======================================================
``sales_timeseries.csv``                 datetime × numeric × category → line chart, with
                                         messy column names, missing values, and duplicates
``iris_like.csv``                        numeric pairs and a low-cardinality label → scatter
                                         with a real correlation, plus correlation heatmap
``survey_responses.csv``                 categorical × categorical → grouped bar / heatmap,
                                         with a boolean column and a free-text field
``sensor_readings.csv``                  a single skewed numeric column → histogram, plus
                                         genuine outliers for the IQR flagger to find
``city_population.json``                 JSON loading and a high-cardinality categorical →
                                         treemap
======================================  ======================================================

Run with ``make samples`` or ``python scripts/make_samples.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
RNG = np.random.default_rng(20260726)


def sales_timeseries() -> pd.DataFrame:
    """Daily revenue per region over two years — the classic time-series case.

    Includes deliberately ugly column names, a missing-value patch, and duplicated rows so the
    cleaning report has something to report.
    """
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    regions = ["EMEA", "North America", "APAC", "LATAM"]
    base = {"EMEA": 42_000, "North America": 68_000, "APAC": 31_000, "LATAM": 18_000}
    growth = {"EMEA": 0.9, "North America": 1.4, "APAC": 1.8, "LATAM": 0.6}

    rows = []
    for region in regions:
        trend = np.linspace(0, growth[region], len(dates))
        seasonal = 0.18 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
        weekly = np.where(pd.Series(dates).dt.dayofweek.isin([5, 6]), -0.25, 0.05)
        noise = RNG.normal(0, 0.07, len(dates))
        revenue = base[region] * (1 + trend + seasonal + weekly + noise)
        units = np.round(revenue / RNG.uniform(38, 52, len(dates))).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "Order Date": dates,
                    "Region ": region,
                    "Total Revenue": np.round(revenue, 2),
                    "unitsSold": units,
                    "Discount %": np.round(RNG.beta(2, 8, len(dates)) * 30, 1),
                }
            )
        )

    df = pd.concat(rows, ignore_index=True)
    # A gap in reporting, so the missing-value strategy has work to do.
    gap = RNG.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[gap, "Total Revenue"] = np.nan
    # Duplicate submissions, so the deduplicator reports something.
    df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)
    return df.sample(frac=1, random_state=2).reset_index(drop=True)


def iris_like() -> pd.DataFrame:
    """Four correlated measurements across three species — scatter and correlation matrix."""
    species_params = {
        "andina": {"n": 60, "length": (5.0, 0.35), "ratio": 0.62, "petal": (1.5, 0.18)},
        "borealis": {"n": 60, "length": (6.0, 0.50), "ratio": 0.47, "petal": (4.3, 0.45)},
        "carinata": {"n": 60, "length": (6.6, 0.62), "ratio": 0.45, "petal": (5.6, 0.55)},
    }
    frames = []
    for species, params in species_params.items():
        n = int(params["n"])
        length = RNG.normal(*params["length"], n)  # type: ignore[misc]
        width = length * params["ratio"] + RNG.normal(0, 0.16, n)
        petal_length = RNG.normal(*params["petal"], n)  # type: ignore[misc]
        petal_width = petal_length * 0.36 + RNG.normal(0, 0.12, n)
        frames.append(
            pd.DataFrame(
                {
                    "sepal_length": np.round(length, 2),
                    "sepal_width": np.round(width, 2),
                    "petal_length": np.round(petal_length, 2),
                    "petal_width": np.round(np.clip(petal_width, 0.05, None), 2),
                    "species": species,
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
    )


def survey_responses() -> pd.DataFrame:
    """A product survey — two categorical dimensions, a boolean, and a free-text column."""
    n = 900
    departments = ["Engineering", "Design", "Sales", "Support", "Finance", "Operations"]
    satisfaction = ["Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"]
    weights = {
        "Engineering": [0.30, 0.38, 0.18, 0.10, 0.04],
        "Design": [0.34, 0.36, 0.18, 0.08, 0.04],
        "Sales": [0.16, 0.28, 0.26, 0.20, 0.10],
        "Support": [0.12, 0.24, 0.28, 0.24, 0.12],
        "Finance": [0.22, 0.34, 0.28, 0.12, 0.04],
        "Operations": [0.20, 0.32, 0.28, 0.14, 0.06],
    }
    comments = [
        "Works well for our team",
        "Needs better reporting",
        "Too slow on large files",
        "Love the export options",
        "Onboarding was confusing",
        "",
    ]

    dept = RNG.choice(departments, n, p=[0.26, 0.10, 0.22, 0.18, 0.10, 0.14])
    rows = {
        "respondent_id": [f"R{i:05d}" for i in range(1, n + 1)],
        "department": dept,
        "satisfaction": [RNG.choice(satisfaction, p=weights[d]) for d in dept],
        "tenure_years": np.round(RNG.gamma(2.0, 1.8, n), 1),
        "would_recommend": RNG.choice([True, False], n, p=[0.71, 0.29]),
        "seat_count": RNG.integers(1, 40, n),
        "comment": RNG.choice(comments, n, p=[0.18, 0.16, 0.14, 0.16, 0.11, 0.25]),
    }
    df = pd.DataFrame(rows)
    df.loc[RNG.choice(df.index, 45, replace=False), "tenure_years"] = np.nan
    return df


def sensor_readings() -> pd.DataFrame:
    """One heavily skewed numeric column with real outliers — histogram and IQR flagging."""
    n = 5_000
    baseline = RNG.lognormal(mean=2.6, sigma=0.45, size=n)
    spikes = RNG.choice(n, size=48, replace=False)
    baseline[spikes] *= RNG.uniform(4.5, 11.0, size=spikes.size)
    return pd.DataFrame(
        {
            "reading_id": np.arange(1, n + 1),
            "particulate_ugm3": np.round(baseline, 2),
            "humidity_pct": np.round(np.clip(RNG.normal(58, 12, n), 5, 99), 1),
            "sensor_status": RNG.choice(["ok", "degraded"], n, p=[0.94, 0.06]),
        }
    )


def city_population() -> list[dict]:
    """Many cities with populations — JSON loading and the treemap path."""
    countries = {
        "Bangladesh": 1.0,
        "India": 1.6,
        "Nigeria": 1.1,
        "Brazil": 0.9,
        "Germany": 0.7,
        "Japan": 0.8,
        "Mexico": 0.9,
        "Egypt": 1.0,
    }
    records = []
    index = 1
    for country, scale in countries.items():
        for city_no in range(1, 13):
            records.append(
                {
                    "city": f"{country[:3].upper()}-City-{city_no:02d}",
                    "country": country,
                    "population": int(RNG.lognormal(13.4, 0.85) * scale),
                    "area_km2": round(float(RNG.uniform(45, 2_400)), 1),
                    "founded": int(RNG.integers(1100, 1985)),
                }
            )
            index += 1
    return records


def main() -> None:
    """Write every sample dataset to ``samples/``."""
    SAMPLES.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, int]] = []

    for name, frame in (
        ("sales_timeseries.csv", sales_timeseries()),
        ("iris_like.csv", iris_like()),
        ("survey_responses.csv", survey_responses()),
        ("sensor_readings.csv", sensor_readings()),
    ):
        path = SAMPLES / name
        frame.to_csv(path, index=False)
        written.append((name, path.stat().st_size))

    cities = city_population()
    path = SAMPLES / "city_population.json"
    path.write_text(json.dumps(cities, indent=2), encoding="utf-8")
    written.append((path.name, path.stat().st_size))

    for name, size in written:
        print(f"  {name:<28} {size / 1024:>8.1f} KB")
    print(f"\nWrote {len(written)} sample datasets to {SAMPLES}")


if __name__ == "__main__":
    main()

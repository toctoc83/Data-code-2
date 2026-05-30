"""Create a compensation bar chart from google_ml_salary_levels.csv.

This script is intentionally simple: it loads the CSV dataset and creates a PNG
chart for the levels where public total-compensation estimates are available.

Run:
    python plot_google_ml_salary_levels.py

Output:
    google_ml_salary_levels_chart.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path("google_ml_salary_levels.csv")
OUTPUT_PATH = Path("google_ml_salary_levels_chart.png")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}. Run this script from the repository root.")

    df = pd.read_csv(CSV_PATH)
    plot_df = df.dropna(subset=["total_comp_usd"]).copy()

    if plot_df.empty:
        raise ValueError("No rows with total compensation values found.")

    plot_df["total_comp_usd"] = plot_df["total_comp_usd"].astype(float)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["level"], plot_df["total_comp_usd"])
    plt.title("Estimated Google ML / AI Total Compensation by Level\nSan Francisco Bay Area / Silicon Valley")
    plt.xlabel("Google level")
    plt.ylabel("Estimated total compensation per year (USD)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)

    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

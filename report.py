import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = Path("models")

stocks = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "JPM"]
algos = ["rf", "gb"]
horizons = [1, 3, 7]

rows = []

for stock in stocks:
    for horizon in horizons:
        for algo in algos:

            path = MODELS_DIR / f"{stock}_{horizon}d_{algo}_meta.json"

            if not path.exists():
                continue

            with open(path, "r") as f:
                meta = json.load(f)

            rows.append({
                "Stock": stock,
                "Horizon": f"{horizon}D",
                "Algorithm": algo.upper(),
                "Accuracy": round(meta["accuracy"] * 100, 2),
                "Precision": round(meta["precision"] * 100, 2),
                "Recall": round(meta["recall"] * 100, 2),
                "MAE": meta["mae"],
                "RMSE": meta["rmse"],
            })

df = pd.DataFrame(rows)

print("\nFULL RESULTS:\n")
print(df)

df.to_csv("evaluation_results.csv", index=False)

print("\nSaved evaluation_results.csv")

avg = (
    df.groupby(["Horizon", "Algorithm"])[
        ["Accuracy", "Precision", "Recall", "MAE", "RMSE"]
    ]
    .mean()
    .round(2)
)

print("\nAVERAGE RESULTS:\n")
print(avg)

avg.to_csv("evaluation_summary.csv")

print("\nSaved evaluation_summary.csv")

plt.figure(figsize=(10, 6))

for algo in ["RF", "GB"]:
    subset = (
        df[df["Algorithm"] == algo]
        .groupby("Horizon")["Accuracy"]
        .mean()
    )

    plt.plot(subset.index, subset.values, marker="o", label=algo)

plt.title("Average Prediction Accuracy by Horizon")
plt.xlabel("Prediction Horizon")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.savefig("accuracy_plot.png")
plt.close()

print("\nSaved accuracy_plot.png")
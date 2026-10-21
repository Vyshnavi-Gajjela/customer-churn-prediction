"""Generate reproducible exploratory plots and data-quality outputs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_eda(df, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    # Target distribution
    counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().reindex(["Stayed", "Churned"])
    fig, ax = plt.subplots(figsize=(6, 4)); counts.plot(kind="bar", ax=ax)
    ax.set(title="Customer churn distribution", xlabel="Outcome", ylabel="Customers"); fig.tight_layout()
    fig.savefig(out/"churn_distribution.png", dpi=150); plt.close(fig)
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        if col not in df: continue
        rates = df.assign(ChurnRate=df["Churn"]).groupby(col, dropna=False)["ChurnRate"].agg(["mean", "count"]).reset_index()
        rates["mean"] *= 100
        fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(rates[col].astype(str), rates["mean"])
        ax.set(title=f"Churn rate by {col}", xlabel=col, ylabel="Churn rate (%)"); ax.tick_params(axis="x", rotation=25)
        for i, row in rates.iterrows(): ax.text(i, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(out/f"churn_rate_by_{col.lower()}.png", dpi=150); plt.close(fig)

"""Render README charts from results/*.json into figures/."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "figures"
C1, C2, C3, MUTED, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#9a9891", "#2b2f2b", "#e1e0d9"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": "#666",
                     "ytick.color": "#666", "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})


def main():
    a = json.loads((HERE / "results" / "analysis.json").read_text())
    m = json.loads((HERE / "results" / "model.json").read_text())
    OUT.mkdir(exist_ok=True)

    yrs = a["by_year"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.bar([y["year"] for y in yrs], [y["funded_usd"] / 1e9 for y in yrs], color=C1, width=0.6)
    ax1.set(ylabel="Funded ($bn)", title="Lending Club volume and approval rate by year")
    ax1.grid(axis="x", visible=False)
    ax2.plot([y["year"] for y in yrs], [y["approval_rate"] * 100 for y in yrs], color=C2, lw=2, marker="o", ms=5)
    ax2.set(ylabel="Applications approved (%)", ylim=(0, None))
    fig.savefig(OUT / "volume_approval.png"); plt.close(fig)

    v = a["vintage"]
    years = sorted({r["year"] for r in v}); grades = sorted({r["grade"] for r in v})
    grid = np.full((len(grades), len(years)), np.nan)
    for r in v:
        grid[grades.index(r["grade"]), years.index(r["year"])] = r["bad_rate"] * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(grid))
    for i in range(len(grades)):
        for j in range(len(years)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if grid[i, j] > np.nanmax(grid) * 0.55 else INK)
    ax.set_xticks(range(len(years)), years); ax.set_yticks(range(len(grades)), grades)
    ax.grid(False)
    ax.set(title="Charge-off rate (%) by vintage and grade, 36-month loans held to maturity", xlabel="Issue year")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(OUT / "vintage_heatmap.png"); plt.close(fig)

    sg = [r for r in a["sub_grade"] if r["loans"] >= 500]
    x = np.arange(len(sg))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax1.plot(x, [r["avg_int_rate"] for r in sg], color=C1, lw=2, label="Interest rate (% a year)")
    ax1.plot(x, [r["bad_rate"] * 100 for r in sg], color=C2, lw=2, label="Loans charged off (% over the 3-year term)")
    ax1.set(ylabel="%", title="Sub-grades with 500+ loans: price vs losses (36-month loans, 2010–2015)")
    ax1.legend(frameon=False)
    ret = [r["pooled_ann_return"] * 100 for r in sg]
    ax2.bar(x, ret, color=[C1 if v >= 0 else C2 for v in ret], width=0.7)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set(ylabel="Realised return (% a year)")
    ax2.set_xticks(x, [r["sub_grade"] for r in sg], rotation=90, fontsize=8)
    ax2.grid(axis="x", visible=False)
    fig.savefig(OUT / "subgrade_economics.png"); plt.close(fig)

    wanted = ["Buy every 2015 36-month loan", "Grade A only", "Highest interest rate, top 20%",
              "Lowest predicted default, top 20%", "Highest predicted return, top 20%"]
    s = [next(x for x in m["strategies"] if x["strategy"] == w) for w in wanted]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh([x["strategy"] for x in s][::-1], [x["pooled_ann_return"] * 100 for x in s][::-1],
                   color=[C1 if "predicted return" in x["strategy"] else MUTED for x in s][::-1], height=0.55)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set(xlabel="Realised annual return (%)", title="Picking 2015 loans with a model trained on 2007–2014")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "strategies.png"); plt.close(fig)

    imp = m["importances"][:10][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.barh([i["feature"] for i in imp], [i["auc_drop"] for i in imp], color=C1, height=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(xlabel="Drop in AUC when shuffled", title="What drives the default model (2015 test loans)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "importance.png"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()

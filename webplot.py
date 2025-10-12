import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
from flask import Flask, Response
from pathlib import Path

app = Flask(__name__)
CSV_FILE = Path(__file__).with_name("results.csv")

# ------------------------------------------------------------------
# 1.  Read the CSV with new structure
# ------------------------------------------------------------------
DF = pd.read_csv(CSV_FILE, parse_dates=["date", "date_pred_short", "date_pred_long"])

# ------------------------------------------------------------------
# 2.  Build the figure
# ------------------------------------------------------------------
def build_plot():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax2 = ax.twinx()

    # --- main curves (using actual trade dates) ---
    ax.plot(DF.date, DF.equity,   label="Strategy equity", color="#1f77b4", linewidth=2)
    ax.plot(DF.date, DF.buyhold,  label="Buy & hold",      color="#ff7f0e", linewidth=2)

    # --- offset & scaled predictions (plotted at their forecast dates) ---
    y_min = min(DF.equity.min(), DF.buyhold.min())
    y_max = max(DF.equity.max(), DF.buyhold.max())
    mid   = (y_max + y_min) / 2                 # vertical midpoint
    cap_range = y_max - y_min
    scale = (cap_range / 2) / 10                # half the range divided by pred max value

    # Plot predictions at their respective forecast target dates
    ax.plot(DF.date_pred_short, mid + DF.pred_short * scale,
            label="Pred Short (7d)", color="#2ca02c", alpha=0.7, linewidth=1.5, 
            linestyle='--', marker='o', markersize=2)
    ax.plot(DF.date_pred_long, mid + DF.pred_long * scale,
            label="Pred Long (25d)", color="#d62728", alpha=0.7, linewidth=1.5,
            linestyle='--', marker='s', markersize=2)

    # --- position histogram (using actual trade dates) ---
    ax2.bar(DF.date, DF.pos, width=1, alpha=0.2, color="grey", label="Position")
    ax2.set_ylabel("Position", fontsize=11)
    ax2.set_ylim(-1.3, 1.3)
    ax2.tick_params(axis='y', labelsize=10)

    # --- styling ---
    ax.set_title("BTC Trading Strategy: Capital & Predictions Over Time", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Value (USD)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, 
             loc="upper left", fontsize=10, framealpha=0.9)
    
    ax.grid(True, ls="--", lw=0.5, alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add annotation about prediction timing
    textstr = 'Note: Predictions plotted at their forecast target dates'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=props)

    # return PNG bytes
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ------------------------------------------------------------------
# 3.  Flask route
# ------------------------------------------------------------------
@app.route("/")
def chart():
    png = build_plot()
    return Response(png, mimetype="image/png")

# ------------------------------------------------------------------
# 4.  Entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

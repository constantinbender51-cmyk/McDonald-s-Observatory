import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
from flask import Flask, Response
from pathlib import Path

app = Flask(__name__)
CSV_FILE = Path(__file__).with_name("results.csv")

# ------------------------------------------------------------------
# 1.  Read the CSV
# ------------------------------------------------------------------
DF = pd.read_csv(CSV_FILE, parse_dates=["date"]).set_index("date")

# ------------------------------------------------------------------
# 2.  Build the figure
# ------------------------------------------------------------------
def build_plot():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    # --- main curves ---
    ax.plot(DF.index, DF.equity,   label="Strategy equity", color="#1f77b4")
    ax.plot(DF.index, DF.buyhold,  label="Buy & hold",      color="#ff7f0e")
    ax.plot(DF.index, DF.pred6,    label="Pred 6d",         color="#2ca02c", alpha=.6)
    ax.plot(DF.index, DF.pred10,   label="Pred 10d",        color="#d62728", alpha=.6)

    # --- position histogram ---
    ax2.bar(DF.index, DF.pos, width=1, alpha=.2, color="grey", label="Position")
    ax2.set_ylabel("Position")
    ax2.set_ylim(-1.2, 1.2)

    ax.set_title("Capital over time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    ax.grid(True, ls="--", lw=.5)

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

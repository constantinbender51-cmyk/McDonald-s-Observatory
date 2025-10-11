import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
from flask import Flask, Response
from pathlib import Path
import io


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
    fig, ax = plt.subplots(figsize=(14, 7))

    # ------------------------------------------------------------------
    # 1.  dollar curves (left axis)
    # ------------------------------------------------------------------
    ax.plot(DF.index, DF.equity,  label='Strategy equity', color='#1f77b4', lw=1.8)
    ax.plot(DF.index, DF.buyhold, label='Buy & hold',      color='#ff7f0e', lw=1.8)
    ax.set_ylabel('Capital ($)', fontsize=11)

    # ------------------------------------------------------------------
    # 2.  centre % forecasts at mid-height of the dollar axis
    # ------------------------------------------------------------------
    y_min, y_max = ax.get_ylim()
    mid_height   = (y_max + y_min) / 2
    half_height  = (y_max - y_min) / 2

    # scale: 30 % should span half the window
    scale = half_height / 30.0
    pred6_centred  = mid_height + DF.pred6  * scale
    pred10_centred = mid_height + DF.pred10 * scale

    ax.plot(DF.index, pred6_centred,  label='Pred 6d (%)',  color='#2ca02c', lw=1.4, alpha=.9)
    ax.plot(DF.index, pred10_centred, label='Pred 10d (%)', color='#d62728', lw=1.4, alpha=.9)

    # ------------------------------------------------------------------
    # 3.  right axis with true % labels
    # ------------------------------------------------------------------
    ax2 = ax.twinx()
    ax2.set_ylim(-30, 30)                       # match the ±30 % range
    ax2.set_ylabel('Forecast (%)', fontsize=11)
    ax2.axhline(0, color='k', lw=.5, ls='--')

    # ------------------------------------------------------------------
    # 4.  faint position bars
    # ------------------------------------------------------------------
    ax.bar(DF.index, DF.pos * half_height * .05,
           width=.8, alpha=.15, color='grey', label='Position')

    # ------------------------------------------------------------------
    # 5.  cosmetics
    # ------------------------------------------------------------------
    ax.set_title('BTC-USD strategy – capital & overlayed forecasts', fontsize=13)
    ax.legend(loc='upper left')
    ax.grid(True, ls='--', lw=.4)

    # return PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
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

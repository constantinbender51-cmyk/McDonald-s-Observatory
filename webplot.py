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
    # 1.  dollar curves (left axis) – nothing special
    # ------------------------------------------------------------------
    ax.plot(DF.index, DF.equity,  label='Strategy equity', color='#1f77b4', lw=1.8)
    ax.plot(DF.index, DF.buyhold, label='Buy & hold',      color='#ff7f0e', lw=1.8)
    ax.set_ylabel('Capital ($)', fontsize=11)
    ax.set_ylim(0, None)

    # ------------------------------------------------------------------
    # 2.  rescale forecasts so they overlay nicely
    # ------------------------------------------------------------------
    idx0   = DF.index[0]
    eq0    = DF.loc[idx0, 'equity']          # first equity value
    scale6 = eq0 / 100                       # 1 %  → 1 scale-unit
    scale10= eq0 / 100

    # convert % forecasts to “indexed” capital lines
    pred6_scaled  = eq0 + DF.pred6  * scale6
    pred10_scaled = eq0 + DF.pred10 * scale10

    ax.plot(DF.index, pred6_scaled,  label='Pred 6d (%)',  color='#2ca02c', lw=1.2, alpha=.8)
    ax.plot(DF.index, pred10_scaled, label='Pred 10d (%)', color='#d62728', lw=1.2, alpha=.8)

    # ------------------------------------------------------------------
    # 3.  fake right axis that shows the original % values
    # ------------------------------------------------------------------
    ax2 = ax.twinx()
    ax2.set_ylim((ax.get_ylim()[0] - eq0) / scale6,
                 (ax.get_ylim()[1] - eq0) / scale6)
    ax2.set_ylabel('Forecast (%)', fontsize=11)
    ax2.axhline(0, color='k', lw=.5, ls='--')

    # ------------------------------------------------------------------
    # 4.  position bars (still on main axis, very faint)
    # ------------------------------------------------------------------
    ax.bar(DF.index, DF.pos * ax.get_ylim()[1] * .03,
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

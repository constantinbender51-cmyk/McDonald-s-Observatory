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
    import io
    fig, ax = plt.subplots(figsize=(14, 7))

    # 1.  capital curves (LEFT axis)
    ax.plot(DF.index, DF.equity,  label='Strategy equity', color='#1f77b4', lw=1.8)
    ax.plot(DF.index, DF.buyhold, label='Buy & hold',      color='#ff7f0e', lw=1.8)
    ax.set_ylabel('Capital ($)', fontsize=11)

    # 2.  prediction curves (RIGHT axis)  –  this keeps auto-scale
    ax2 = ax.twinx()
    ax2.plot(DF.index, DF.pred6,  label='Pred 6d (%)',  color='#2ca02c', lw=1.4, alpha=.9)
    ax2.plot(DF.index, DF.pred10, label='Pred 10d (%)', color='#d62728', lw=1.4, alpha=.9)
    ax2.axhline(0, color='k', lw=.6, ls='--')
    ax2.set_ylabel('Forecast (%)', fontsize=11)

    # 3.  position bars – full height on LEFT axis
    ax.bar(DF.index, DF.pos, width=.8, alpha=.25, color='grey', label='Position')
    ax.set_ylim(-1.2, 1.2)          # -1 / 0 / +1
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['short', 'flat', 'long'])

    # 4.  cosmetics
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

import re
import io
import base64
from datetime import datetime
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
from flask import Flask, Response

app = Flask(__name__)
LOG_FILE = "logs.1760194374372.log.txt"   # <- same folder or absolute path

# ------------------------------------------------------------------
# 1.  Parse the log once at start-up (cheap for a demo)
# ------------------------------------------------------------------
def parse_log(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            # 2025-10-11T14:42:04.446080053Z [inf]  2024-02-07    1.2   -0.1    0   1000.00    953.24
            m = re.search(r"(\d{4}-\d{2}-\d{2})\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if not m:
                continue
            date, p6, p10, pos, eq, bh = m.groups()
            rows.append({
                "date": datetime.strptime(date, "%Y-%m-%d"),
                "pred6": float(p6),
                "pred10": float(p10),
                "pos": int(float(pos)),
                "equity": float(eq),
                "buyhold": float(bh)
            })
    return pd.DataFrame(rows).set_index("date")

DF = parse_log(LOG_FILE)

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
    buf = io.BytesIO()
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
# 4.  Railway entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Railway injects PORT env-var
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

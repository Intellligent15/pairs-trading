# ETF Pairs Mean‑Reversion — Web App

This package wraps your existing pairs‑trading project into a simple **Streamlit** web app that you can deploy quickly.

## Option A — Streamlit Community Cloud (fastest)

1. Push this project (including `app.py` and `requirements.txt`) to a public GitHub repo.
2. Go to Streamlit Community Cloud → New app → select the repo/branch.
3. Set the main file to `app.py` and deploy.

### Using your code
Open **app.py** and find the function `run_with_your_code(...)`.
Replace its body with calls into your modules (e.g., `backtest`, `main`, etc.) and return a dict like:
```python
{
  "summary": {
      "cagr": 0.12, "sharpe": 1.5, "max_drawdown": -0.08, "trades": 34, "win_rate": 0.56, "exposure": 0.48
  },
  "equity_df": pd.DataFrame(..., columns=["equity"]).set_index("Date"),
  "signals_df": pd.DataFrame(..., columns=["z","exec_signal","qx","qy"]).set_index("Date")
}
```
Until you wire your functions, the app runs a **fallback demo** using free `yfinance` data.

## Option B — Render (free tier) or Railway

- Ensure this folder contains `Procfile` and `requirements.txt`.
- Create a new **Web Service**:
  - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Add any **environment variables** your code needs (instead of committing secrets).

## Option C — Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```
Build & run:
```bash
docker build -t etf-pairs-app .
docker run -p 8501:8501 etf-pairs-app
```

## Secrets & API Keys

- Do **not** commit secrets. If your project uses `keys.py`, refactor to read from environment variables:
```python
import os
API_KEY = os.getenv("API_KEY")
```
- Set these env vars in Streamlit Cloud / Render dashboard.

## Local dev

```bash
pip install -r requirements.txt
streamlit run app.py
```

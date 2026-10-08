import pandas as pd

COLUMNS = ["date", "ticker", "side", "shares", "price", "strategy", "notes"]
CLOSED_COLUMNS = ["date", "ticker", "shares", "avg_cost", "exit_price", "pnl", "return_pct", "strategy"]
OPEN_COLUMNS = ["ticker", "shares", "avg_cost"]


def empty_journal():
    return pd.DataFrame(columns=COLUMNS)


def clean_journal(df):
    """Normalize a journal loaded from a CSV or built in the app."""
    missing = [c for c in ["date", "ticker", "side", "shares", "price"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    df = df.copy()
    for col in ["strategy", "notes"]:
        if col not in df.columns:
            df[col] = ""
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"])
    df["price"] = pd.to_numeric(df["price"])
    df["strategy"] = df["strategy"].fillna("").astype(str)
    df["notes"] = df["notes"].fillna("").astype(str)
    bad = df[~df["side"].isin(["BUY", "SELL"])]
    if len(bad):
        raise ValueError("Side must be BUY or SELL")
    return df[COLUMNS].reset_index(drop=True)


def add_trade(df, date, ticker, side, shares, price, strategy="", notes=""):
    row = pd.DataFrame([{
        "date": date, "ticker": ticker, "side": side, "shares": shares,
        "price": price, "strategy": strategy, "notes": notes,
    }])
    combined = row if df.empty else pd.concat([df, row], ignore_index=True)
    return clean_journal(combined)


def compute_positions(trades):
    """
    Replay trades in date order using the average cost method.
    Returns (open positions, closed trades with realized P&L).
    Sells larger than the current position are capped at what you hold.
    """
    if trades.empty:
        return pd.DataFrame(columns=OPEN_COLUMNS), pd.DataFrame(columns=CLOSED_COLUMNS)

    book = {}
    closed = []
    for row in trades.sort_values("date", kind="stable").itertuples():
        pos = book.setdefault(row.ticker, {"shares": 0.0, "cost": 0.0})
        if row.side == "BUY":
            pos["shares"] += row.shares
            pos["cost"] += row.shares * row.price
            continue

        qty = min(row.shares, pos["shares"])
        if qty <= 0:
            continue
        avg = pos["cost"] / pos["shares"]
        closed.append({
            "date": row.date,
            "ticker": row.ticker,
            "shares": qty,
            "avg_cost": avg,
            "exit_price": row.price,
            "pnl": (row.price - avg) * qty,
            "return_pct": (row.price / avg - 1) * 100,
            "strategy": row.strategy,
        })
        pos["shares"] -= qty
        pos["cost"] -= avg * qty

    open_rows = [
        {"ticker": t, "shares": p["shares"], "avg_cost": p["cost"] / p["shares"]}
        for t, p in book.items() if p["shares"] > 1e-9
    ]
    return pd.DataFrame(open_rows, columns=OPEN_COLUMNS), pd.DataFrame(closed, columns=CLOSED_COLUMNS)


def performance_stats(closed):
    if closed.empty:
        return None
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] < 0]
    gross_loss = -losses["pnl"].sum()
    return {
        "total_pnl": closed["pnl"].sum(),
        "num_trades": len(closed),
        "win_rate": len(wins) / len(closed) * 100,
        "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,
        "profit_factor": wins["pnl"].sum() / gross_loss if gross_loss > 0 else float("inf"),
        "best": closed["pnl"].max(),
        "worst": closed["pnl"].min(),
    }


def stats_by_strategy(closed):
    if closed.empty:
        return pd.DataFrame()
    df = closed.assign(strategy=closed["strategy"].replace("", "Untagged"))
    g = df.groupby("strategy")
    return pd.DataFrame({
        "trades": g.size(),
        "win_rate": g["pnl"].apply(lambda s: (s > 0).mean() * 100),
        "total_pnl": g["pnl"].sum(),
        "avg_return_pct": g["return_pct"].mean(),
    }).sort_values("total_pnl", ascending=False)


def sample_journal():
    rows = [
        ("2025-01-06", "AAPL", "BUY", 20, 243.0, "Breakout", "Bought after it cleared resistance"),
        ("2025-01-13", "NVDA", "BUY", 15, 133.0, "Earnings", "Positioning before earnings season"),
        ("2025-02-03", "MSFT", "BUY", 10, 410.0, "Dip buy", "Down 6% on the week"),
        ("2025-02-18", "AAPL", "SELL", 10, 244.5, "Breakout", "Trimmed half, momentum fading"),
        ("2025-03-10", "NVDA", "SELL", 15, 107.0, "Earnings", "Stopped out, should have sized smaller"),
        ("2025-04-09", "AMZN", "BUY", 12, 191.0, "Dip buy", "Market selloff, bought the panic"),
        ("2025-05-12", "AMZN", "SELL", 12, 208.0, "Dip buy", "Took profit into strength"),
        ("2025-06-02", "TSLA", "BUY", 8, 342.0, "Momentum", ""),
        ("2025-06-20", "TSLA", "SELL", 8, 322.0, "Momentum", "Broke the 20 day average"),
        ("2025-07-14", "META", "BUY", 6, 718.0, "Breakout", "New highs on volume"),
    ]
    return clean_journal(pd.DataFrame(rows, columns=COLUMNS))

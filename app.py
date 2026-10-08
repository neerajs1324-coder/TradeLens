from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from backtest import extract_trades, metrics, rsi_signal, run_backtest, sma_crossover_signal
from journal import (
    add_trade, clean_journal, compute_positions, empty_journal,
    performance_stats, sample_journal, stats_by_strategy,
)

st.set_page_config(page_title="TradeLens", page_icon="📈", layout="wide")

GREEN, RED, BLUE, GRAY = "#16A34A", "#DC2626", "#2563EB", "#94A3B8"
STRATEGIES = ["Breakout", "Dip buy", "Earnings", "Momentum", "Swing", "Long term", "Other"]


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker, start, end):
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return pd.Series(dtype=float)
    close = hist["Close"]
    close.index = close.index.tz_localize(None)
    return close


@st.cache_data(ttl=300, show_spinner=False)
def latest_prices(tickers):
    out = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="5d")
            if not hist.empty:
                out[t] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return out


def money(x):
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def style_chart(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
    )
    return fig


if "trades" not in st.session_state:
    st.session_state.trades = empty_journal()


# sidebar

with st.sidebar:
    st.title("📈 TradeLens")
    st.caption("A trading journal and strategy backtester.")

    st.subheader("Your data")
    st.caption("Trades live in this session. Download a CSV to keep them, then upload it next time.")

    if st.button("Load sample trades", width="stretch"):
        st.session_state.trades = sample_journal()
        st.rerun()

    uploaded = st.file_uploader("Upload journal CSV", type="csv")
    if uploaded is not None and st.session_state.get("last_upload") != uploaded.file_id:
        try:
            st.session_state.trades = clean_journal(pd.read_csv(uploaded))
            st.session_state.last_upload = uploaded.file_id
            st.success("Journal loaded")
        except Exception as e:
            st.error(f"Couldn't read that file. {e}")

    st.download_button(
        "Download journal CSV",
        st.session_state.trades.to_csv(index=False).encode(),
        file_name="trade_journal.csv",
        mime="text/csv",
        width="stretch",
        disabled=st.session_state.trades.empty,
    )

    if st.button("Clear all trades", width="stretch", disabled=st.session_state.trades.empty):
        st.session_state.trades = empty_journal()
        st.rerun()


trades = st.session_state.trades
open_pos, closed = compute_positions(trades)

tab_journal, tab_portfolio, tab_perf, tab_backtest = st.tabs(["Journal", "Portfolio", "Performance", "Backtester"])


# journal

with tab_journal:
    st.header("Trade journal")

    with st.form("new_trade", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 0.8, 1, 1])
        t_date = c1.date_input("Date", value=date.today(), max_value=date.today())
        t_ticker = c2.text_input("Ticker", placeholder="AAPL")
        t_side = c3.selectbox("Side", ["BUY", "SELL"])
        t_shares = c4.number_input("Shares", min_value=0.0, step=1.0, value=None, placeholder="10")
        t_price = c5.number_input("Price", min_value=0.0, step=0.01, value=None, placeholder="150.00")
        c6, c7 = st.columns([1, 3])
        t_strategy = c6.selectbox("Strategy", STRATEGIES)
        t_notes = c7.text_input("Notes", placeholder="Why did you take this trade?")

        if st.form_submit_button("Add trade", type="primary"):
            if not t_ticker.strip() or not t_shares or not t_price:
                st.error("Fill in the ticker, shares, and price.")
            else:
                st.session_state.trades = add_trade(
                    trades, t_date, t_ticker, t_side, t_shares, t_price, t_strategy, t_notes
                )
                st.rerun()

    if trades.empty:
        st.info("No trades yet. Add one above, or load the sample trades from the sidebar.")
    else:
        view = trades.sort_values("date", ascending=False)
        st.dataframe(
            view,
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                "ticker": "Ticker",
                "side": "Side",
                "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "strategy": "Strategy",
                "notes": st.column_config.TextColumn("Notes", width="large"),
            },
        )

        with st.expander("Delete trades"):
            labels = {
                i: f"{r.date:%b %d, %Y}  {r.side} {r.shares:g} {r.ticker} @ ${r.price:,.2f}"
                for i, r in trades.iterrows()
            }
            to_delete = st.multiselect("Pick trades to delete", list(labels), format_func=labels.get)
            if st.button("Delete selected", disabled=not to_delete):
                st.session_state.trades = trades.drop(index=to_delete).reset_index(drop=True)
                st.rerun()


# portfolio

with tab_portfolio:
    st.header("Open positions")

    if open_pos.empty:
        st.info("You don't have any open positions. Buy trades you haven't sold show up here.")
    else:
        with st.spinner("Getting live prices"):
            quotes = latest_prices(tuple(open_pos["ticker"]))

        port = open_pos.copy()
        port["price"] = port["ticker"].map(quotes)
        port["cost_basis"] = port["shares"] * port["avg_cost"]
        port["market_value"] = port["shares"] * port["price"]
        port["unrealized_pnl"] = port["market_value"] - port["cost_basis"]
        port["unrealized_pct"] = port["unrealized_pnl"] / port["cost_basis"] * 100

        total_value = port["market_value"].sum()
        total_cost = port.loc[port["price"].notna(), "cost_basis"].sum()
        total_unreal = port["unrealized_pnl"].sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Market value", money(total_value))
        m2.metric("Cost basis", money(port["cost_basis"].sum()))
        m3.metric(
            "Unrealized P&L", money(total_unreal),
            f"{total_unreal / total_cost * 100:+.2f}%" if total_cost else None,
        )

        if port["price"].isna().any():
            missing = ", ".join(port.loc[port["price"].isna(), "ticker"])
            st.warning(f"Couldn't get a live price for {missing}. Check the ticker symbol.")

        left, right = st.columns([3, 2])
        with left:
            st.dataframe(
                port,
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                    "avg_cost": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "cost_basis": st.column_config.NumberColumn("Cost basis", format="$%.2f"),
                    "market_value": st.column_config.NumberColumn("Value", format="$%.2f"),
                    "unrealized_pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
                    "unrealized_pct": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
                },
            )
        with right:
            priced = port.dropna(subset=["market_value"])
            if not priced.empty:
                fig = go.Figure(go.Pie(labels=priced["ticker"], values=priced["market_value"], hole=0.55))
                fig.update_layout(title="Allocation", height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig)


# performance

with tab_perf:
    st.header("Performance")
    stats = performance_stats(closed)

    if stats is None:
        st.info("Stats show up once you've closed at least one trade by selling shares you bought.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Realized P&L", money(stats["total_pnl"]))
        m2.metric("Win rate", f"{stats['win_rate']:.0f}%", f"{stats['num_trades']} closed trades", delta_color="off")
        pf = stats["profit_factor"]
        m3.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}",
                  help="Total won divided by total lost. Above 1 means you're making money.")
        m4.metric("Avg win vs loss", f"{money(stats['avg_win'])} / {money(stats['avg_loss'])}")

        curve = closed.sort_values("date").assign(cumulative=lambda d: d["pnl"].cumsum())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["date"], y=curve["cumulative"], mode="lines+markers",
                                 name="Cumulative P&L", line=dict(color=BLUE, width=2.5)))
        fig.add_hline(y=0, line_color=GRAY, line_dash="dot")
        fig.update_layout(title="Cumulative realized P&L", yaxis_tickprefix="$")
        st.plotly_chart(style_chart(fig))

        left, right = st.columns(2)
        with left:
            st.subheader("By strategy")
            st.caption("Which setups are actually working for you.")
            st.dataframe(
                stats_by_strategy(closed),
                column_config={
                    "trades": "Trades",
                    "win_rate": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
                    "total_pnl": st.column_config.NumberColumn("Total P&L", format="$%.2f"),
                    "avg_return_pct": st.column_config.NumberColumn("Avg return", format="%.2f%%"),
                },
            )
        with right:
            st.subheader("Closed trades")
            st.caption(f"Best {money(stats['best'])}, worst {money(stats['worst'])}")
            st.dataframe(
                closed.sort_values("date", ascending=False),
                hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Closed", format="MMM D, YYYY"),
                    "ticker": "Ticker",
                    "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                    "avg_cost": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
                    "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
                    "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
                    "return_pct": st.column_config.NumberColumn("Return", format="%.2f%%"),
                    "strategy": "Strategy",
                },
            )


# backtester

with tab_backtest:
    st.header("Strategy backtester")
    st.caption("Test a trading rule on historical prices and compare it to just buying and holding.")

    c1, c2, c3 = st.columns(3)
    bt_ticker = c1.text_input("Ticker", value="SPY").upper().strip()
    bt_start = c2.date_input("Start", value=date.today() - timedelta(days=365 * 5), max_value=date.today())
    bt_end = c3.date_input("End", value=date.today(), max_value=date.today())

    c4, c5, c6 = st.columns(3)
    bt_strategy = c4.selectbox("Strategy", ["Moving average crossover", "RSI mean reversion"])
    bt_cash = c5.number_input("Starting cash", min_value=100, value=10_000, step=1000)
    bt_fee = c6.number_input("Cost per trade (%)", min_value=0.0, max_value=5.0, value=0.05, step=0.05)

    if bt_strategy == "Moving average crossover":
        st.caption("Holds the stock while the fast average is above the slow average, sits in cash otherwise.")
        p1, p2 = st.columns(2)
        fast = p1.slider("Fast average (days)", 5, 100, 20)
        slow = p2.slider("Slow average (days)", 20, 300, 50)
    else:
        st.caption("Buys when RSI says the stock is oversold, sells when it says overbought.")
        p1, p2, p3 = st.columns(3)
        rsi_period = p1.slider("RSI period", 5, 30, 14)
        buy_below = p2.slider("Buy below", 10, 50, 30)
        sell_above = p3.slider("Sell above", 50, 90, 70)

    if st.button("Run backtest", type="primary"):
        if bt_start >= bt_end:
            st.error("The start date needs to be before the end date.")
        elif bt_strategy == "Moving average crossover" and fast >= slow:
            st.error("The fast average needs to be shorter than the slow one.")
        else:
            with st.spinner(f"Downloading {bt_ticker} prices"):
                close = load_prices(bt_ticker, bt_start, bt_end)
            if len(close) < 30:
                st.error(f"Not enough price data for {bt_ticker}. Check the ticker or pick a longer date range.")
            else:
                if bt_strategy == "Moving average crossover":
                    signal = sma_crossover_signal(close, fast, slow)
                else:
                    signal = rsi_signal(close, rsi_period, buy_below, sell_above)
                st.session_state.bt = {
                    "ticker": bt_ticker,
                    "result": run_backtest(close, signal, bt_cash, bt_fee / 100),
                }

    if "bt" in st.session_state:
        res = st.session_state.bt["result"]
        ticker = st.session_state.bt["ticker"]
        s, b = metrics(res["strategy"]), metrics(res["buy_hold"])
        bt_trades = extract_trades(res)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Final value", money(s["final_value"]), f"{s['final_value'] - b['final_value']:+,.0f} vs buy and hold")
        m2.metric("Total return", f"{s['total_return']:.1f}%", f"{s['total_return'] - b['total_return']:+.1f} pts")
        m3.metric("Annual return", f"{s['cagr']:.1f}%", f"{s['cagr'] - b['cagr']:+.1f} pts")
        m4.metric("Sharpe ratio", f"{s['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f}",
                  help="Return per unit of risk. Higher is better, above 1 is solid.")
        m5.metric("Max drawdown", f"{s['max_drawdown']:.1f}%", f"{s['max_drawdown'] - b['max_drawdown']:+.1f} pts",
                  help="The biggest drop from a peak. Closer to zero is better.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.index, y=res["strategy"], name="Strategy", line=dict(color=BLUE, width=2.5)))
        fig.add_trace(go.Scatter(x=res.index, y=res["buy_hold"], name="Buy and hold", line=dict(color=GRAY, width=2)))
        fig.update_layout(title="Portfolio value", yaxis_tickprefix="$")
        st.plotly_chart(style_chart(fig))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=res.index, y=res["close"], name=ticker, line=dict(color=GRAY, width=1.5)))
        if not bt_trades.empty:
            fig2.add_trace(go.Scatter(x=bt_trades["entry_date"], y=bt_trades["entry_price"], mode="markers",
                                      name="Buy", marker=dict(symbol="triangle-up", size=11, color=GREEN)))
            done = bt_trades[bt_trades["status"] == "Closed"]
            fig2.add_trace(go.Scatter(x=done["exit_date"], y=done["exit_price"], mode="markers",
                                      name="Sell", marker=dict(symbol="triangle-down", size=11, color=RED)))
        fig2.update_layout(title=f"{ticker} price with trades", yaxis_tickprefix="$")
        st.plotly_chart(style_chart(fig2, 340))

        st.subheader(f"{len(bt_trades)} trades")
        if not bt_trades.empty:
            wins = (bt_trades["return_pct"] > 0).mean() * 100
            st.caption(f"{wins:.0f}% were winners, average return {bt_trades['return_pct'].mean():.2f}% per trade")
            st.dataframe(
                bt_trades,
                hide_index=True,
                column_config={
                    "entry_date": st.column_config.DateColumn("Bought", format="MMM D, YYYY"),
                    "entry_price": st.column_config.NumberColumn("Buy price", format="$%.2f"),
                    "exit_date": st.column_config.DateColumn("Sold", format="MMM D, YYYY"),
                    "exit_price": st.column_config.NumberColumn("Sell price", format="$%.2f"),
                    "return_pct": st.column_config.NumberColumn("Return", format="%.2f%%"),
                    "status": "Status",
                },
            )

        st.caption("Past performance doesn't predict future results. This is a learning tool, not financial advice.")

import pandas as pd
import pytest

from journal import add_trade, clean_journal, compute_positions, empty_journal, performance_stats


def make(rows):
    return clean_journal(pd.DataFrame(rows, columns=["date", "ticker", "side", "shares", "price"]))


def test_average_cost_and_realized_pnl():
    trades = make([
        ("2025-01-01", "AAPL", "BUY", 10, 100),
        ("2025-01-02", "AAPL", "BUY", 10, 120),
        ("2025-01-03", "AAPL", "SELL", 5, 130),
    ])
    open_pos, closed = compute_positions(trades)
    assert closed.iloc[0]["avg_cost"] == pytest.approx(110)
    assert closed.iloc[0]["pnl"] == pytest.approx(100)
    assert open_pos.iloc[0]["shares"] == pytest.approx(15)
    assert open_pos.iloc[0]["avg_cost"] == pytest.approx(110)


def test_fully_closed_position_disappears():
    trades = make([
        ("2025-01-01", "TSLA", "BUY", 4, 200),
        ("2025-01-05", "TSLA", "SELL", 4, 180),
    ])
    open_pos, closed = compute_positions(trades)
    assert open_pos.empty
    assert closed.iloc[0]["pnl"] == pytest.approx(-80)


def test_oversell_is_capped():
    trades = make([
        ("2025-01-01", "MSFT", "BUY", 2, 50),
        ("2025-01-02", "MSFT", "SELL", 10, 60),
    ])
    _, closed = compute_positions(trades)
    assert closed.iloc[0]["shares"] == 2


def test_trades_replay_in_date_order():
    trades = make([
        ("2025-01-05", "NVDA", "SELL", 5, 150),
        ("2025-01-01", "NVDA", "BUY", 5, 100),
    ])
    _, closed = compute_positions(trades)
    assert closed.iloc[0]["pnl"] == pytest.approx(250)


def test_win_rate():
    trades = make([
        ("2025-01-01", "A", "BUY", 1, 10),
        ("2025-01-02", "A", "SELL", 1, 12),
        ("2025-01-03", "B", "BUY", 1, 10),
        ("2025-01-04", "B", "SELL", 1, 8),
    ])
    _, closed = compute_positions(trades)
    stats = performance_stats(closed)
    assert stats["win_rate"] == pytest.approx(50)
    assert stats["total_pnl"] == pytest.approx(0)


def test_add_trade_uppercases_ticker():
    df = add_trade(empty_journal(), "2025-01-01", " aapl ", "buy", 1, 100)
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["side"] == "BUY"


def test_bad_side_rejected():
    with pytest.raises(ValueError):
        make([("2025-01-01", "AAPL", "HOLD", 1, 100)])

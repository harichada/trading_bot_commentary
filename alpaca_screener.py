from alpaca.data.screener import Screener

client = Screener()

# Get gap up stocks
gap_ups = client.query(
    rank_by="gap_up_percent",
    limit=50,
    filters={
        "min_price": 2.0,
        "min_volume": 1_000_000,
        "gap_threshold": 2.5  # 2.5% gap
    }
)

# Results include: symbol, gap_percent, open, close, volume
for stock in gap_ups:
    print(f"{stock['symbol']}: {stock['gap_up_percent']}%")

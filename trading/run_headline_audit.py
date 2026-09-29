import yfinance as yf
from trading.config import WATCHLIST, YF_SUFFIX
from trading.indian_market_data import filter_curated_news

print("=" * 80)
print("  NEWS HEADLINE FILTERING AUDIT TRAIL ACROSS 6 WATCHLIST SYMBOLS")
print("=" * 80)

for sym in WATCHLIST:
    t = yf.Ticker(sym + YF_SUFFIX)
    raw = getattr(t, "news", []) or []
    filtered, summary, audit = filter_curated_news(raw, sym, max_items=10)
    print(f"\n--- {sym} (Raw headlines: {len(raw)} | Accepted: {len(filtered)}) ---")
    if not audit:
        print("  (No headlines returned from data feed)")
    for entry in audit:
        status_tag = "[ACCEPT]" if entry["status"] == "ACCEPTED" else "[REJECT]"
        print(f"  {status_tag:<8} | Rule: {entry['rule']}")
        print(f"           Headline: \"{entry['headline']}\" ({entry['source']})")

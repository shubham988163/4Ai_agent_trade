"""Quick launcher for the AI Trading Agents Scanner.

Usage:
  python run_ai_trader.py                     # Scan core watchlist
  python run_ai_trader.py --symbol INFY       # Analyze specific symbol
  python run_ai_trader.py --all               # Scan full Nifty-50
  python run_ai_trader.py --min-conviction 6  # Lower conviction threshold to 6/10
"""
from trading.agents.agent_trader import main

if __name__ == "__main__":
    main()

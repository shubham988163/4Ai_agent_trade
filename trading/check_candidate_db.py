import sqlite3, json, sys

conn = sqlite3.connect("data/ledger.db")
conn.row_factory = sqlite3.Row

print("=== CANDIDATE SIGNALS IN data/ledger.db ===")
try:
    rows = conn.execute("SELECT * FROM candidate_signals ORDER BY id DESC LIMIT 10").fetchall()
    print(f"Total candidate rows: {len(rows)}")
    for r in rows:
        d = dict(r)
        print(f"ID={d.get('id')} | Date={d.get('date')} | Symbol={d.get('symbol')} | Side={d.get('side')} | Executed={d.get('executed')} | Action={d.get('council_action')} | Conviction={d.get('council_conviction')}")
        print(f"  Reasons: {d.get('council_reasons')}")
        print(f"  Facts: {d.get('facts_json')[:100] if d.get('facts_json') else None}...")
except Exception as e:
    print(f"Error querying candidate_signals: {e}")

print("\n=== AGENT LOG CALLS FOR SBIN ===")
try:
    rows = conn.execute("SELECT * FROM agent_log WHERE prompt LIKE '%SBIN%' ORDER BY id DESC LIMIT 5").fetchall()
    print(f"Total SBIN calls in agent_log: {len(rows)}")
    for r in rows:
        d = dict(r)
        print(f"ID={d.get('id')} | Agent={d.get('agent')} | Model={d.get('model')} | OK={d.get('ok')}")
        print(f"  Prompt snippet: {d.get('prompt')[:200]}...")
        print(f"  Response: {d.get('response')}")
except Exception as e:
    print(f"Error querying agent_log: {e}")

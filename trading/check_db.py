import sqlite3, glob

dbs = glob.glob("**/*.db", recursive=True)
print("DBs found:", dbs)
for db in dbs:
    try:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print(f"\n=== {db} (tables: {tables}) ===")
        if "candidate_signals" in tables:
            rows = conn.execute("SELECT * FROM candidate_signals").fetchall()
            print(f"candidate_signals count: {len(rows)}")
            for r in rows:
                print("Candidate row:", dict(r))
        if "agent_log" in tables:
            rows = conn.execute("SELECT * FROM agent_log").fetchall()
            print(f"agent_log count: {len(rows)}")
            for r in rows:
                print("Agent log row:", dict(r))
    except Exception as e:
        print(f"Error {db}: {e}")

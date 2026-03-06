"""
Phase 1: Database Setup Script
Creates policy_analyser database and policy_user, verifies connectivity.
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

SUPERUSER = "postgres"
SUPERPASS = "admin"
HOST = "localhost"
PORT = 5432

TARGET_DB = "policy_analyser"
TARGET_USER = "policy_user"
TARGET_PASS = "admin"


def main():
    # 1. Connect as superuser
    print("[1] Connecting to PostgreSQL as superuser...")
    conn = psycopg2.connect(host=HOST, port=PORT, user=SUPERUSER, password=SUPERPASS, dbname="postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    print("    ✓ Connected to PostgreSQL")

    # 2. Create database if not exists
    print(f"[2] Creating database '{TARGET_DB}'...")
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{TARGET_DB}'")
    if cur.fetchone():
        print(f"    ✓ Database '{TARGET_DB}' already exists")
    else:
        cur.execute(f"CREATE DATABASE {TARGET_DB}")
        print(f"    ✓ Database '{TARGET_DB}' created")

    # 3. Create user if not exists
    print(f"[3] Creating user '{TARGET_USER}'...")
    cur.execute(f"SELECT 1 FROM pg_roles WHERE rolname = '{TARGET_USER}'")
    if cur.fetchone():
        print(f"    ✓ User '{TARGET_USER}' already exists")
        cur.execute(f"ALTER USER {TARGET_USER} WITH PASSWORD '{TARGET_PASS}'")
    else:
        cur.execute(f"CREATE USER {TARGET_USER} WITH PASSWORD '{TARGET_PASS}'")
        print(f"    ✓ User '{TARGET_USER}' created")

    # 4. Grant privileges
    print(f"[4] Granting privileges...")
    cur.execute(f"GRANT ALL PRIVILEGES ON DATABASE {TARGET_DB} TO {TARGET_USER}")
    print(f"    ✓ Privileges granted on '{TARGET_DB}'")

    cur.close()
    conn.close()

    # 5. Connect to target DB as superuser to grant schema permissions
    print("[5] Granting schema permissions...")
    conn2 = psycopg2.connect(host=HOST, port=PORT, user=SUPERUSER, password=SUPERPASS, dbname=TARGET_DB)
    conn2.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur2 = conn2.cursor()
    cur2.execute(f"GRANT ALL ON SCHEMA public TO {TARGET_USER}")
    cur2.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {TARGET_USER}")
    cur2.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {TARGET_USER}")
    # Ensure uuid-ossp extension exists
    cur2.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
    print("    ✓ Schema permissions + uuid-ossp extension OK")
    cur2.close()
    conn2.close()

    # 6. Verify connection as target user
    print(f"[6] Verifying connection as '{TARGET_USER}'...")
    try:
        conn3 = psycopg2.connect(host=HOST, port=PORT, user=TARGET_USER, password=TARGET_PASS, dbname=TARGET_DB)
        cur3 = conn3.cursor()
        cur3.execute("SELECT current_user, current_database(), version()")
        user, db, ver = cur3.fetchone()
        print(f"    ✓ Connected as '{user}' to '{db}'")
        print(f"    ✓ PostgreSQL: {ver[:60]}...")
        cur3.close()
        conn3.close()
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        raise

    print("\n═══ Phase 1 COMPLETE ═══")


if __name__ == "__main__":
    main()

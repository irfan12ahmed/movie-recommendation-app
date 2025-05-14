# tests/test_auth.py
from auth import hash_password, authenticate_user, add_user, init_db
import os
import sqlite3

def test_hash_password():
    assert hash_password("password123") == hash_password("password123")

def test_add_and_authenticate_user(tmp_path):
    db_path = tmp_path / "test_users.db"
    os.environ["TEST_DB"] = str(db_path)
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("CREATE TABLE users (username TEXT PRIMARY KEY, password TEXT)")
    conn.commit()
    conn.close()

    add_user("testuser", "testpass")
    assert authenticate_user("testuser", "testpass") is True
    assert authenticate_user("testuser", "wrongpass") is False
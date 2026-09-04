"""
AyurHerb - Database Management Module (database.py)
Author: Data Science Team
Description: SQLite-based search logging and history tracking for the AyurHerb application.
             Stores query audit records without collecting sensitive personal medical data.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional


def get_db_path() -> str:
    """Determine absolute path to the SQLite database file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(base_dir, "instance")
    os.makedirs(instance_dir, exist_ok=True)
    return os.path.join(instance_dir, "ayurherb.db")


def get_db_connection() -> sqlite3.Connection:
    """Establish and return SQLite connection with row factory enabled."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize SQLite tables if they do not exist.
    Called automatically upon Flask application startup.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            top_condition TEXT,
            similarity_score REAL,
            results_count INTEGER DEFAULT 0,
            safety_triggered INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print(f"AyurHerb database initialized at: {get_db_path()}")


def log_search(
    query: str,
    top_condition: str = "None",
    similarity_score: float = 0.0,
    results_count: int = 0,
    safety_triggered: bool = False
) -> int:
    """
    Log an anonymized symptom search event into the SQLite database.
    """
    if not query or not query.strip():
        return -1
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO search_history (query, timestamp, top_condition, similarity_score, results_count, safety_triggered)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            query.strip(),
            now_str,
            top_condition or "None",
            float(similarity_score),
            int(results_count),
            1 if safety_triggered else 0
        ))
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id
    except Exception as e:
        print(f"Error logging search query: {e}")
        return -1


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve past search queries ordered by timestamp descending.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, query, timestamp, top_condition, similarity_score, results_count, safety_triggered
            FROM search_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        history = [dict(row) for row in rows]
        conn.close()
        return history
    except Exception as e:
        print(f"Error fetching search history: {e}")
        return []


def clear_history() -> bool:
    """
    Clear all records from search_history table.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_history")
        conn.commit()
        conn.close()
        print("Search history cleared successfully.")
        return True
    except Exception as e:
        print(f"Error clearing search history: {e}")
        return False


if __name__ == "__main__":
    # Test database functions
    init_db()
    log_search("cough and sore throat", "Cough", 0.675, 3, False)
    log_search("severe chest pain", "Emergency Triggered", 0.0, 0, True)
    h = get_history()
    print(f"History records ({len(h)}):")
    for r in h:
        print(" ", r)
    print("Database test passed!")

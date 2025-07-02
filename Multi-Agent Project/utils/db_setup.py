# multi_agent_bot/utils/db_setup.py
import sqlite3
import os
from datetime import datetime

DATABASE_FILE = 'leave_management.db'

def init_db():
    """Initializes the SQLite database with necessary tables."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Create employees table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            annual_leave_balance INTEGER DEFAULT 20,
            sick_leave_balance INTEGER DEFAULT 10
        )
    ''')

    # Create leave_requests table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leave_requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'Pending',
            requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        )
    ''')

    # # Insert some dummy employee data if tables are empty
    # cursor.execute("INSERT OR IGNORE INTO employees (employee_id, name, annual_leave_balance, sick_leave_balance) VALUES (?, ?, ?, ?)",
    #                ('EMP001', 'Aditya Patadiya', 15, 5))
    # cursor.execute("INSERT OR IGNORE INTO employees (employee_id, name, annual_leave_balance, sick_leave_balance) VALUES (?, ?, ?, ?)",
    #                ('EMP002', 'Jane Doe', 10, 3))

    conn.commit()
    conn.close()
    print(f"Database '{DATABASE_FILE}' initialized and populated (if empty).")

if __name__ == '__main__':
    init_db()
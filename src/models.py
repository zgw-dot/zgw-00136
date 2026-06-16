import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "app.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            labeled_count INTEGER NOT NULL,
            label_distribution TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            columns_info TEXT NOT NULL,
            created_at TEXT NOT NULL,
            note TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL UNIQUE,
            dataset_version TEXT NOT NULL,
            model_path TEXT NOT NULL,
            vectorizer_path TEXT NOT NULL,
            label_encoder_path TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 0,
            training_status TEXT NOT NULL,
            error_message TEXT,
            hyperparams TEXT NOT NULL,
            feature_count INTEGER,
            created_at TEXT NOT NULL,
            training_completed_at TEXT,
            note TEXT,
            FOREIGN KEY (dataset_version) REFERENCES dataset_versions(version)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            accuracy REAL NOT NULL,
            precision_macro REAL NOT NULL,
            recall_macro REAL NOT NULL,
            f1_macro REAL NOT NULL,
            report_json TEXT NOT NULL,
            confusion_matrix TEXT NOT NULL,
            labels TEXT NOT NULL,
            train_samples INTEGER NOT NULL,
            test_samples INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (model_version) REFERENCES model_versions(version),
            FOREIGN KEY (dataset_version) REFERENCES dataset_versions(version)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            clause_text TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            corrected_label TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (model_version) REFERENCES model_versions(version)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS rollback_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            previous_active_version TEXT,
            new_active_version TEXT,
            operation_type TEXT NOT NULL,
            operator_note TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def now_iso() -> str:
    return datetime.now().isoformat()


def dict_factory(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def parse_json_field(field: Optional[str]) -> Any:
    if field is None:
        return None
    try:
        return json.loads(field)
    except (json.JSONDecodeError, TypeError):
        return field

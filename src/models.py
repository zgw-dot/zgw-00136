import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "app.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()


def ensure_permissions(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    default_perms = [
        ("admin", "batch_view", 1),
        ("admin", "batch_create", 1),
        ("admin", "batch_export", 1),
        ("admin", "batch_audit_export", 1),
        ("admin", "correction_create", 1),
        ("admin", "correction_revert", 1),
        ("admin", "model_train", 1),
        ("admin", "model_activate", 1),
        ("admin", "dataset_import", 1),
        ("admin", "filter_scheme_manage", 1),
        ("reviewer", "batch_view", 1),
        ("reviewer", "batch_export", 1),
        ("reviewer", "batch_audit_export", 1),
        ("reviewer", "correction_create", 1),
        ("reviewer", "correction_revert", 1),
        ("reviewer", "filter_scheme_manage", 1),
        ("viewer", "batch_view", 1),
        ("viewer", "batch_export", 1),
        ("viewer", "batch_audit_export", 0),
        ("viewer", "filter_scheme_manage", 0),
    ]
    for role, op, allowed in default_perms:
        cur.execute(
            "SELECT allowed FROM permissions WHERE role = ? AND operation = ?",
            (role, op),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO permissions (role, operation, allowed, created_at) VALUES (?, ?, ?, ?)",
                (role, op, allowed, now_iso()),
            )
    conn.commit()


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
            note TEXT,
            source_type TEXT DEFAULT 'manual_upload',
            operator TEXT
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
            operator TEXT,
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
            operator TEXT,
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
            operator TEXT,
            is_reverted INTEGER NOT NULL DEFAULT 0,
            reverted_at TEXT,
            reverted_by TEXT,
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
            created_at TEXT NOT NULL,
            operator TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS batch_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            model_version TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            total_rows INTEGER NOT NULL DEFAULT 0,
            predicted_count INTEGER NOT NULL DEFAULT 0,
            conflict_count INTEGER NOT NULL DEFAULT 0,
            corrected_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'completed',
            note TEXT,
            created_at TEXT NOT NULL,
            source_type TEXT DEFAULT 'manual_upload',
            operator TEXT,
            FOREIGN KEY (model_version) REFERENCES model_versions(version),
            FOREIGN KEY (dataset_version) REFERENCES dataset_versions(version)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS batch_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            clause_text TEXT NOT NULL,
            contract_type TEXT,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            top_predictions TEXT NOT NULL,
            is_conflict INTEGER NOT NULL DEFAULT 0,
            conflict_reason TEXT,
            true_label TEXT,
            corrected_label TEXT,
            correction_reason TEXT,
            correction_id INTEGER,
            previous_label TEXT,
            previous_reason TEXT,
            previous_operator TEXT,
            previous_corrected_at TEXT,
            created_at TEXT NOT NULL,
            corrected_at TEXT,
            corrected_by TEXT,
            FOREIGN KEY (batch_id) REFERENCES batch_predictions(batch_id),
            FOREIGN KEY (correction_id) REFERENCES corrections(id)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_batch_items_batch_id ON batch_items(batch_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS export_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            export_id TEXT NOT NULL UNIQUE,
            batch_id TEXT,
            export_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            model_version TEXT,
            item_count INTEGER NOT NULL DEFAULT 0,
            note TEXT,
            created_at TEXT NOT NULL,
            operator TEXT,
            FOREIGN KEY (batch_id) REFERENCES batch_predictions(batch_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_type TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            details TEXT,
            operator TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_op_logs_type ON operation_logs(operation_type)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_op_logs_entity ON operation_logs(entity_type, entity_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS correction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            correction_id INTEGER NOT NULL,
            batch_id TEXT,
            batch_item_id INTEGER,
            model_version TEXT NOT NULL,
            clause_text TEXT NOT NULL,
            previous_label TEXT,
            previous_reason TEXT,
            previous_operator TEXT,
            previous_corrected_at TEXT,
            new_label TEXT NOT NULL,
            new_reason TEXT NOT NULL,
            new_operator TEXT,
            operation_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (correction_id) REFERENCES corrections(id),
            FOREIGN KEY (batch_id) REFERENCES batch_predictions(batch_id)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_corr_hist_corr ON correction_history(correction_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_corr_hist_batch ON correction_history(batch_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT NOT NULL UNIQUE,
            batch_id TEXT NOT NULL,
            applicant TEXT NOT NULL,
            applicant_role TEXT NOT NULL DEFAULT 'viewer',
            status TEXT NOT NULL DEFAULT 'active',
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            revoked_at TEXT,
            revoked_by TEXT,
            reissued_from INTEGER,
            last_download_at TEXT,
            download_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (batch_id) REFERENCES batch_predictions(batch_id),
            FOREIGN KEY (reissued_from) REFERENCES audit_tokens(id)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_tokens_token ON audit_tokens(token)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_tokens_batch ON audit_tokens(batch_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_tokens_status ON audit_tokens(status)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_share_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id INTEGER,
            batch_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            operator TEXT,
            applicant TEXT,
            details TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (token_id) REFERENCES audit_tokens(id),
            FOREIGN KEY (batch_id) REFERENCES batch_predictions(batch_id)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_share_logs_batch ON audit_share_logs(batch_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_share_logs_token ON audit_share_logs(token_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_share_logs_event ON audit_share_logs(event_type)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS filter_schemes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            owner TEXT NOT NULL DEFAULT 'default',
            is_default INTEGER NOT NULL DEFAULT 0,
            model_version TEXT,
            dataset_version TEXT,
            created_from TEXT,
            created_to TEXT,
            source_type TEXT,
            has_conflicts TEXT,
            has_corrections TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(name, owner)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_filter_schemes_owner ON filter_schemes(owner)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_filter_schemes_default ON filter_schemes(owner, is_default)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS permissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            operation TEXT NOT NULL,
            allowed INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            UNIQUE(role, operation)
        )
    """)

    _ensure_column(conn, "dataset_versions", "source_type", "TEXT DEFAULT 'manual_upload'")
    _ensure_column(conn, "dataset_versions", "operator", "TEXT")
    _ensure_column(conn, "model_versions", "operator", "TEXT")
    _ensure_column(conn, "evaluations", "operator", "TEXT")
    _ensure_column(conn, "corrections", "operator", "TEXT")
    _ensure_column(conn, "corrections", "is_reverted", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "corrections", "reverted_at", "TEXT")
    _ensure_column(conn, "corrections", "reverted_by", "TEXT")
    _ensure_column(conn, "rollback_logs", "operator", "TEXT")
    _ensure_column(conn, "batch_predictions", "corrected_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "batch_predictions", "source_type", "TEXT DEFAULT 'manual_upload'")
    _ensure_column(conn, "batch_predictions", "operator", "TEXT")
    _ensure_column(conn, "batch_items", "previous_label", "TEXT")
    _ensure_column(conn, "batch_items", "previous_reason", "TEXT")
    _ensure_column(conn, "batch_items", "previous_operator", "TEXT")
    _ensure_column(conn, "batch_items", "previous_corrected_at", "TEXT")
    _ensure_column(conn, "batch_items", "corrected_by", "TEXT")
    _ensure_column(conn, "export_files", "operator", "TEXT")

    cur.execute("SELECT COUNT(*) FROM permissions")
    if cur.fetchone()[0] == 0:
        default_perms = [
            ("admin", "batch_view", 1),
            ("admin", "batch_create", 1),
            ("admin", "batch_export", 1),
            ("admin", "batch_audit_export", 1),
            ("admin", "correction_create", 1),
            ("admin", "correction_revert", 1),
            ("admin", "model_train", 1),
            ("admin", "model_activate", 1),
            ("admin", "dataset_import", 1),
            ("admin", "filter_scheme_manage", 1),
            ("reviewer", "batch_view", 1),
            ("reviewer", "batch_export", 1),
            ("reviewer", "batch_audit_export", 1),
            ("reviewer", "correction_create", 1),
            ("reviewer", "correction_revert", 1),
            ("reviewer", "filter_scheme_manage", 1),
            ("viewer", "batch_view", 1),
            ("viewer", "batch_export", 1),
            ("viewer", "batch_audit_export", 0),
            ("viewer", "filter_scheme_manage", 0),
        ]
        for role, op, allowed in default_perms:
            cur.execute(
                "INSERT INTO permissions (role, operation, allowed, created_at) VALUES (?, ?, ?, ?)",
                (role, op, allowed, now_iso()),
            )
        conn.commit()
    else:
        ensure_permissions(conn)

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

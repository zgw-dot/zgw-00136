import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from .models import get_connection, now_iso, parse_json_field, dict_factory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "data", "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
EVALUATIONS_DIR = os.path.join(BASE_DIR, "data", "evaluations")


def _gen_dataset_version() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"ds_{ts}_{uuid.uuid4().hex[:6]}"


def _gen_model_version() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"mdl_{ts}_{uuid.uuid4().hex[:6]}"


def compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_dataset_version(
    filename: str,
    source_path: str,
    row_count: int,
    labeled_count: int,
    label_distribution: Dict[str, int],
    columns_info: List[str],
    note: Optional[str] = None,
) -> str:
    version = _gen_dataset_version()
    file_hash = compute_file_hash(source_path)

    stored_path = os.path.join(DATASETS_DIR, f"{version}.csv")
    shutil.copy2(source_path, stored_path)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO dataset_versions
        (version, filename, row_count, labeled_count, label_distribution,
         file_hash, columns_info, created_at, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version,
            filename,
            row_count,
            labeled_count,
            json.dumps(label_distribution, ensure_ascii=False),
            file_hash,
            json.dumps(columns_info, ensure_ascii=False),
            now_iso(),
            note,
        ),
    )
    conn.commit()
    conn.close()
    return version


def list_dataset_versions() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM dataset_versions ORDER BY created_at DESC")
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["label_distribution"] = parse_json_field(r["label_distribution"])
        r["columns_info"] = parse_json_field(r["columns_info"])
        r["stored_path"] = os.path.join(DATASETS_DIR, f"{r['version']}.csv")
    return rows


def get_dataset_version(version: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM dataset_versions WHERE version = ?", (version,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["label_distribution"] = parse_json_field(r["label_distribution"])
    r["columns_info"] = parse_json_field(r["columns_info"])
    r["stored_path"] = os.path.join(DATASETS_DIR, f"{r['version']}.csv")
    return r


def create_model_version_entry(
    dataset_version: str,
    hyperparams: Dict[str, Any],
    note: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    version = _gen_model_version()
    model_dir = os.path.join(MODELS_DIR, version)
    os.makedirs(model_dir, exist_ok=True)

    paths = {
        "model_path": os.path.join(model_dir, "classifier.joblib"),
        "vectorizer_path": os.path.join(model_dir, "vectorizer.joblib"),
        "label_encoder_path": os.path.join(model_dir, "label_encoder.joblib"),
    }

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_versions
        (version, dataset_version, model_path, vectorizer_path, label_encoder_path,
         is_active, training_status, error_message, hyperparams, created_at, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version,
            dataset_version,
            paths["model_path"],
            paths["vectorizer_path"],
            paths["label_encoder_path"],
            0,
            "training",
            None,
            json.dumps(hyperparams, ensure_ascii=False),
            now_iso(),
            note,
        ),
    )
    conn.commit()
    conn.close()
    return version, paths


def update_model_version_status(
    version: str,
    status: str,
    error_message: Optional[str] = None,
    feature_count: Optional[int] = None,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    if status == "completed":
        cur.execute(
            """
            UPDATE model_versions
            SET training_status = ?, error_message = ?, feature_count = ?,
                training_completed_at = ?
            WHERE version = ?
            """,
            (status, error_message, feature_count, now_iso(), version),
        )
    else:
        cur.execute(
            """
            UPDATE model_versions
            SET training_status = ?, error_message = ?, feature_count = ?
            WHERE version = ?
            """,
            (status, error_message, feature_count, version),
        )
    conn.commit()
    conn.close()


def activate_model_version(version: str, operator_note: Optional[str] = None) -> bool:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT version FROM model_versions WHERE is_active = 1 AND training_status = 'completed'"
    )
    prev = cur.fetchone()
    prev_version = prev["version"] if prev else None

    cur.execute(
        "SELECT training_status FROM model_versions WHERE version = ?", (version,)
    )
    target = cur.fetchone()
    if target is None:
        conn.close()
        return False
    if target["training_status"] != "completed":
        conn.close()
        return False

    cur.execute("UPDATE model_versions SET is_active = 0 WHERE is_active = 1")
    cur.execute("UPDATE model_versions SET is_active = 1 WHERE version = ?", (version,))

    cur.execute(
        """
        INSERT INTO rollback_logs
        (previous_active_version, new_active_version, operation_type, operator_note, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (prev_version, version, "activate", operator_note, now_iso()),
    )

    conn.commit()
    conn.close()
    return True


def list_model_versions() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["hyperparams"] = parse_json_field(r["hyperparams"])
        r["is_active"] = bool(r["is_active"])
    return rows


def get_active_model_version() -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM model_versions WHERE is_active = 1 AND training_status = 'completed' LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["hyperparams"] = parse_json_field(r["hyperparams"])
    r["is_active"] = bool(r["is_active"])
    return r


def get_model_version(version: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_versions WHERE version = ?", (version,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["hyperparams"] = parse_json_field(r["hyperparams"])
    r["is_active"] = bool(r["is_active"])
    return r


def add_correction(
    model_version: str,
    clause_text: str,
    predicted_label: str,
    corrected_label: str,
    reason: str,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO corrections
        (model_version, clause_text, predicted_label, corrected_label, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            model_version,
            clause_text,
            predicted_label,
            corrected_label,
            reason,
            now_iso(),
        ),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def list_corrections(model_version: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    if model_version:
        cur.execute(
            "SELECT * FROM corrections WHERE model_version = ? ORDER BY created_at DESC",
            (model_version,),
        )
    else:
        cur.execute("SELECT * FROM corrections ORDER BY created_at DESC")
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    return rows


def list_rollback_logs() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM rollback_logs ORDER BY created_at DESC")
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_evaluation(
    model_version: str,
    dataset_version: str,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    report_json: Dict[str, Any],
    confusion_matrix: List[List[int]],
    labels: List[str],
    train_samples: int,
    test_samples: int,
) -> Tuple[int, str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO evaluations
        (model_version, dataset_version, accuracy, precision_macro, recall_macro,
         f1_macro, report_json, confusion_matrix, labels, train_samples, test_samples, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_version,
            dataset_version,
            accuracy,
            precision_macro,
            recall_macro,
            f1_macro,
            json.dumps(report_json, ensure_ascii=False),
            json.dumps(confusion_matrix, ensure_ascii=False),
            json.dumps(labels, ensure_ascii=False),
            train_samples,
            test_samples,
            now_iso(),
        ),
    )
    eval_id = cur.lastrowid
    conn.commit()
    conn.close()

    report_path = os.path.join(EVALUATIONS_DIR, f"eval_{eval_id}_{model_version}.json")
    full_report = {
        "evaluation_id": eval_id,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "metrics": {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        },
        "report": report_json,
        "confusion_matrix": confusion_matrix,
        "labels": labels,
        "split": {"train_samples": train_samples, "test_samples": test_samples},
        "generated_at": now_iso(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    return eval_id, report_path


def list_evaluations(model_version: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    if model_version:
        cur.execute(
            "SELECT * FROM evaluations WHERE model_version = ? ORDER BY created_at DESC",
            (model_version,),
        )
    else:
        cur.execute("SELECT * FROM evaluations ORDER BY created_at DESC")
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["report_json"] = parse_json_field(r["report_json"])
        r["confusion_matrix"] = parse_json_field(r["confusion_matrix"])
        r["labels"] = parse_json_field(r["labels"])
    return rows


def get_evaluation(eval_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["report_json"] = parse_json_field(r["report_json"])
    r["confusion_matrix"] = parse_json_field(r["confusion_matrix"])
    r["labels"] = parse_json_field(r["labels"])
    return r


def add_operation_log(
    operation_type: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    operator: Optional[str] = None,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO operation_logs
        (operation_type, entity_type, entity_id, details, operator, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            operation_type,
            entity_type,
            entity_id,
            json.dumps(details, ensure_ascii=False) if details else None,
            operator,
            now_iso(),
        ),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def list_operation_logs(
    operation_type: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM operation_logs WHERE 1=1"
    params: List[Any] = []
    if operation_type:
        query += " AND operation_type = ?"
        params.append(operation_type)
    if entity_type:
        query += " AND entity_type = ?"
        params.append(entity_type)
    if entity_id:
        query += " AND entity_id = ?"
        params.append(entity_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["details"] = parse_json_field(r["details"])
    return rows


def add_correction_history(
    correction_id: int,
    batch_id: Optional[str],
    batch_item_id: Optional[int],
    model_version: str,
    clause_text: str,
    previous_label: Optional[str],
    previous_reason: Optional[str],
    previous_operator: Optional[str],
    previous_corrected_at: Optional[str],
    new_label: str,
    new_reason: str,
    new_operator: Optional[str],
    operation_type: str,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO correction_history
        (correction_id, batch_id, batch_item_id, model_version, clause_text,
         previous_label, previous_reason, previous_operator, previous_corrected_at,
         new_label, new_reason, new_operator, operation_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            correction_id,
            batch_id,
            batch_item_id,
            model_version,
            clause_text,
            previous_label,
            previous_reason,
            previous_operator,
            previous_corrected_at,
            new_label,
            new_reason,
            new_operator,
            operation_type,
            now_iso(),
        ),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def list_correction_history(
    correction_id: Optional[int] = None,
    batch_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM correction_history WHERE 1=1"
    params: List[Any] = []
    if correction_id is not None:
        query += " AND correction_id = ?"
        params.append(correction_id)
    if batch_id:
        query += " AND batch_id = ?"
        params.append(batch_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    return rows


def check_permission(role: str, operation: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT allowed FROM permissions WHERE role = ? AND operation = ?",
        (role, operation),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return False
    return bool(row["allowed"])


def list_permissions() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM permissions ORDER BY role, operation")
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["allowed"] = bool(r["allowed"])
    return rows


def get_correction(correction_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["is_reverted"] = bool(r.get("is_reverted", 0))
    return r


def mark_correction_reverted(correction_id: int, operator: Optional[str] = None) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE corrections
        SET is_reverted = 1, reverted_at = ?, reverted_by = ?
        WHERE id = ? AND is_reverted = 0
        """,
        (now_iso(), operator, correction_id),
    )
    updated = cur.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def list_filter_schemes(owner: str = "default") -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM filter_schemes WHERE owner = ? ORDER BY is_default DESC, updated_at DESC",
        (owner,),
    )
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["is_default"] = bool(r["is_default"])
    return rows


def get_filter_scheme(scheme_id: int, owner: str = "default") -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM filter_schemes WHERE id = ? AND owner = ?",
        (scheme_id, owner),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["is_default"] = bool(r["is_default"])
    return r


def get_default_filter_scheme(owner: str = "default") -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM filter_schemes WHERE owner = ? AND is_default = 1 LIMIT 1",
        (owner,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["is_default"] = bool(r["is_default"])
    return r


def create_filter_scheme(
    name: str,
    owner: str = "default",
    is_default: bool = False,
    model_version: Optional[str] = None,
    dataset_version: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    source_type: Optional[str] = None,
    has_conflicts: Optional[str] = None,
    has_corrections: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()

    try:
        if is_default:
            cur.execute(
                "UPDATE filter_schemes SET is_default = 0, updated_at = ? WHERE owner = ? AND is_default = 1",
                (now_iso(), owner),
            )

        cur.execute(
            """
            INSERT INTO filter_schemes
            (name, owner, is_default, model_version, dataset_version,
             created_from, created_to, source_type, has_conflicts,
             has_corrections, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                owner,
                1 if is_default else 0,
                model_version,
                dataset_version,
                created_from,
                created_to,
                source_type,
                has_conflicts,
                has_corrections,
                now_iso(),
                now_iso(),
            ),
        )
        scheme_id = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"筛选方案名称 '{name}' 已存在，请使用其他名称")

    conn.close()

    result = get_filter_scheme(scheme_id, owner)
    if result is None:
        raise RuntimeError("创建筛选方案失败")
    return result


def update_filter_scheme(
    scheme_id: int,
    owner: str = "default",
    name: Optional[str] = None,
    is_default: Optional[bool] = None,
    model_version: Optional[str] = None,
    dataset_version: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    source_type: Optional[str] = None,
    has_conflicts: Optional[str] = None,
    has_corrections: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()

    existing = get_filter_scheme(scheme_id, owner)
    if existing is None:
        conn.close()
        raise ValueError(f"筛选方案不存在: {scheme_id}")

    try:
        updates: List[str] = []
        params: List[Any] = []

        if name is not None and name != existing["name"]:
            updates.append("name = ?")
            params.append(name)
        if is_default is not None:
            updates.append("is_default = ?")
            params.append(1 if is_default else 0)
        if model_version is not None:
            updates.append("model_version = ?")
            params.append(model_version)
        if dataset_version is not None:
            updates.append("dataset_version = ?")
            params.append(dataset_version)
        if created_from is not None:
            updates.append("created_from = ?")
            params.append(created_from)
        if created_to is not None:
            updates.append("created_to = ?")
            params.append(created_to)
        if source_type is not None:
            updates.append("source_type = ?")
            params.append(source_type)
        if has_conflicts is not None:
            updates.append("has_conflicts = ?")
            params.append(has_conflicts)
        if has_corrections is not None:
            updates.append("has_corrections = ?")
            params.append(has_corrections)

        if not updates:
            conn.close()
            return existing

        updates.append("updated_at = ?")
        params.append(now_iso())
        params.append(scheme_id)
        params.append(owner)

        if is_default:
            cur.execute(
                "UPDATE filter_schemes SET is_default = 0, updated_at = ? WHERE owner = ? AND is_default = 1 AND id != ?",
                (now_iso(), owner, scheme_id),
            )

        cur.execute(
            f"UPDATE filter_schemes SET {', '.join(updates)} WHERE id = ? AND owner = ?",
            params,
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"筛选方案名称 '{name}' 已存在，请使用其他名称")

    conn.close()

    result = get_filter_scheme(scheme_id, owner)
    if result is None:
        raise RuntimeError("更新筛选方案失败")
    return result


def set_default_filter_scheme(scheme_id: int, owner: str = "default") -> bool:
    conn = get_connection()
    cur = conn.cursor()

    existing = get_filter_scheme(scheme_id, owner)
    if existing is None:
        conn.close()
        return False

    cur.execute(
        "UPDATE filter_schemes SET is_default = 0, updated_at = ? WHERE owner = ? AND is_default = 1",
        (now_iso(), owner),
    )
    cur.execute(
        "UPDATE filter_schemes SET is_default = 1, updated_at = ? WHERE id = ? AND owner = ?",
        (now_iso(), scheme_id, owner),
    )
    conn.commit()
    conn.close()
    return True


def delete_filter_scheme(scheme_id: int, owner: str = "default") -> bool:
    conn = get_connection()
    cur = conn.cursor()

    existing = get_filter_scheme(scheme_id, owner)
    if existing is None:
        conn.close()
        return False

    was_default = existing["is_default"]

    cur.execute(
        "DELETE FROM filter_schemes WHERE id = ? AND owner = ?",
        (scheme_id, owner),
    )
    conn.commit()
    conn.close()

    return True

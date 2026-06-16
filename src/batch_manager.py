import json
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from .models import get_connection, now_iso, dict_factory, parse_json_field
from .version_manager import (
    get_model_version,
    get_active_model_version,
    add_operation_log,
    list_correction_history,
    list_operation_logs,
    check_permission,
)
from .predictor import load_model_bundle, PredictionError
from .trainer import _build_features_from_dataframe


BATCHES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "batches",
)
EXPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "exports",
)

os.makedirs(BATCHES_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)


class BatchError(Exception):
    pass


def _gen_batch_id() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"batch_{ts}_{uuid.uuid4().hex[:6]}"


def _gen_export_id() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"exp_{ts}_{uuid.uuid4().hex[:6]}"


def create_batch_prediction(
    csv_path: str,
    filename: str,
    model_version: Optional[str] = None,
    note: Optional[str] = None,
    top_k: int = 3,
    source_type: str = "manual_upload",
) -> Dict[str, Any]:
    if model_version:
        mv = get_model_version(model_version)
    else:
        mv = get_active_model_version()

    if mv is None:
        raise BatchError(
            f"模型版本不存在: {model_version}" if model_version else "尚无激活的模型"
        )
    if mv["training_status"] != "completed":
        raise BatchError(f"模型 {mv['version']} 训练未完成，无法预测")

    model_version = mv["version"]
    dataset_version = mv["dataset_version"]

    try:
        bundle = load_model_bundle(model_version)
    except PredictionError as e:
        raise BatchError(str(e)) from e

    try:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="gbk")
    except Exception as e:
        raise BatchError(f"读取CSV失败: {e}") from e

    if "条款文本" not in df.columns:
        raise BatchError("CSV缺少必填列: 条款文本")

    df = df[df["条款文本"].astype(str).str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    total_rows = len(df)
    if total_rows == 0:
        raise BatchError("CSV中没有有效的条款文本")

    batch_id = _gen_batch_id()

    seen_texts: Dict[str, int] = {}
    conflict_count = 0
    items: List[Dict[str, Any]] = []

    texts_for_pred = []
    pred_indices = []

    for i, row in df.iterrows():
        text = str(row["条款文本"]).strip()
        contract_type = str(row.get("合同类型", "")) if "合同类型" in df.columns else ""
        true_label = str(row.get("风险标签", "")) if "风险标签" in df.columns else ""

        if text in seen_texts:
            conflict_count += 1
            items.append({
                "row_index": i,
                "clause_text": text,
                "contract_type": contract_type,
                "true_label": true_label,
                "is_conflict": True,
                "conflict_reason": f"与第 {seen_texts[text] + 1} 行条款重复",
                "predicted_label": "",
                "confidence": 0.0,
                "top_predictions": [],
            })
        else:
            seen_texts[text] = i
            pred_indices.append(i)
            texts_for_pred.append(_build_text_feature(text, contract_type))
            items.append({
                "row_index": i,
                "clause_text": text,
                "contract_type": contract_type,
                "true_label": true_label,
                "is_conflict": False,
                "conflict_reason": None,
                "predicted_label": "",
                "confidence": 0.0,
                "top_predictions": [],
            })

    if texts_for_pred:
        import numpy as np
        X = bundle.vectorizer.transform(texts_for_pred)
        probs_all = bundle.classifier.predict_proba(X)
        pred_indices_arr = np.argmax(probs_all, axis=1)
        pred_labels = bundle.label_encoder.inverse_transform(pred_indices_arr)

        for idx_in_pred, orig_idx in enumerate(pred_indices):
            probs = probs_all[idx_in_pred]
            pred_idx = int(pred_indices_arr[idx_in_pred])
            top_inds = np.argsort(probs)[::-1][:top_k]
            top_results = []
            for j in top_inds:
                top_results.append({
                    "label": bundle.label_encoder.inverse_transform([int(j)])[0],
                    "probability": float(probs[j]),
                })

            items[orig_idx]["predicted_label"] = pred_labels[idx_in_pred]
            items[orig_idx]["confidence"] = float(probs[pred_idx])
            items[orig_idx]["top_predictions"] = top_results

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO batch_predictions
            (batch_id, filename, model_version, dataset_version,
             total_rows, predicted_count, conflict_count, status, note, created_at,
             source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                filename,
                model_version,
                dataset_version,
                total_rows,
                total_rows - conflict_count,
                conflict_count,
                "completed",
                note,
                now_iso(),
                source_type,
            ),
        )

        for item in items:
            cur.execute(
                """
                INSERT INTO batch_items
                (batch_id, row_index, clause_text, contract_type,
                 predicted_label, confidence, top_predictions,
                 is_conflict, conflict_reason, true_label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    item["row_index"],
                    item["clause_text"],
                    item["contract_type"],
                    item["predicted_label"],
                    item["confidence"],
                    json.dumps(item["top_predictions"], ensure_ascii=False),
                    1 if item["is_conflict"] else 0,
                    item["conflict_reason"],
                    item["true_label"],
                    now_iso(),
                ),
            )

        conn.commit()
    finally:
        conn.close()

    add_operation_log(
        operation_type="batch_predict_create",
        entity_type="batch",
        entity_id=batch_id,
        details={
            "filename": filename,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "total_rows": total_rows,
            "predicted_count": total_rows - conflict_count,
            "conflict_count": conflict_count,
        },
    )

    return {
        "batch_id": batch_id,
        "filename": filename,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "total_rows": total_rows,
        "predicted_count": total_rows - conflict_count,
        "conflict_count": conflict_count,
        "status": "completed",
        "note": note,
        "created_at": now_iso(),
    }


def _build_text_feature(clause_text: str, contract_type: Optional[str]) -> str:
    if contract_type and str(contract_type).strip():
        return f"[{str(contract_type).strip()}] {clause_text}"
    return clause_text


def list_batches(
    model_version: Optional[str] = None,
    dataset_version: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    source_type: Optional[str] = None,
    has_conflicts: Optional[str] = None,
    has_corrections: Optional[str] = None,
    limit: int = 50,
    role: Optional[str] = "admin",
) -> List[Dict[str, Any]]:
    if role and not check_permission(role, "batch_view"):
        raise BatchError(f"角色 '{role}' 没有查看批次的权限")

    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM batch_predictions WHERE 1=1"
    params: List[Any] = []
    if model_version:
        query += " AND model_version = ?"
        params.append(model_version)
    if dataset_version:
        query += " AND dataset_version = ?"
        params.append(dataset_version)
    if created_from:
        query += " AND created_at >= ?"
        params.append(created_from)
    if created_to:
        query += " AND created_at <= ?"
        params.append(created_to)
    if source_type:
        query += " AND source_type = ?"
        params.append(source_type)
    if has_conflicts == "yes":
        query += " AND conflict_count > 0"
    elif has_conflicts == "no":
        query += " AND conflict_count = 0"
    if has_corrections == "yes":
        query += " AND corrected_count > 0"
    elif has_corrections == "no":
        query += " AND corrected_count = 0"
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_batch_detail(batch_id: str, role: Optional[str] = "admin") -> Optional[Dict[str, Any]]:
    if role and not check_permission(role, "batch_view"):
        raise BatchError(f"角色 '{role}' 没有查看批次的权限")

    batch = get_batch(batch_id)
    if batch is None:
        return None

    counts = get_batch_item_count(batch_id)
    exports = list_exports(batch_id=batch_id, limit=100)
    correction_hist = list_correction_history(batch_id=batch_id, limit=100)
    batch_logs = list_operation_logs(entity_type="batch", entity_id=batch_id, limit=100)
    all_corr_logs = list_operation_logs(entity_type="correction", limit=500)
    corr_logs_for_batch = [
        l for l in all_corr_logs
        if (l.get("details") or {}).get("batch_id") == batch_id
    ]
    op_logs = sorted(
        batch_logs + corr_logs_for_batch,
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )[:100]

    conflict_items = get_batch_items(batch_id, include_conflicts=True, limit=1000)
    conflict_list = [
        {
            "id": c["id"],
            "row_index": c["row_index"],
            "clause_text": c["clause_text"],
            "conflict_reason": c.get("conflict_reason"),
        }
        for c in conflict_items
        if c.get("is_conflict")
    ]

    corrected_items = [
        {
            "id": c["id"],
            "row_index": c["row_index"],
            "clause_text": c["clause_text"],
            "predicted_label": c["predicted_label"],
            "corrected_label": c.get("corrected_label"),
            "correction_reason": c.get("correction_reason"),
            "corrected_by": c.get("corrected_by"),
            "corrected_at": c.get("corrected_at"),
            "previous_label": c.get("previous_label"),
        }
        for c in conflict_items
        if c.get("corrected_label")
    ]

    model_info = get_model_version(batch["model_version"]) if batch.get("model_version") else None

    return {
        **batch,
        "item_counts": counts,
        "exports": exports,
        "correction_history": correction_hist,
        "operation_logs": op_logs,
        "conflict_items": conflict_list,
        "corrected_items_summary": corrected_items,
        "model_info": {
            "version": model_info["version"] if model_info else None,
            "note": model_info.get("note") if model_info else None,
            "created_at": model_info.get("created_at") if model_info else None,
        },
    }


def get_batch(batch_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM batch_predictions WHERE batch_id = ?", (batch_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return dict_factory(row)


def get_batch_items(
    batch_id: str,
    include_conflicts: bool = True,
    limit: int = 1000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM batch_items WHERE batch_id = ?"
    params: List[Any] = [batch_id]
    if not include_conflicts:
        query += " AND is_conflict = 0"
    query += " ORDER BY row_index ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["top_predictions"] = parse_json_field(r["top_predictions"])
        r["is_conflict"] = bool(r["is_conflict"])
    return rows


def get_batch_item_count(batch_id: str) -> Dict[str, int]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) as total FROM batch_items WHERE batch_id = ?",
        (batch_id,),
    )
    total = cur.fetchone()["total"]
    cur.execute(
        "SELECT COUNT(*) as predicted FROM batch_items WHERE batch_id = ? AND is_conflict = 0",
        (batch_id,),
    )
    predicted = cur.fetchone()["predicted"]
    cur.execute(
        "SELECT COUNT(*) as conflicts FROM batch_items WHERE batch_id = ? AND is_conflict = 1",
        (batch_id,),
    )
    conflicts = cur.fetchone()["conflicts"]
    cur.execute(
        "SELECT COUNT(*) as corrected FROM batch_items WHERE batch_id = ? AND corrected_label IS NOT NULL",
        (batch_id,),
    )
    corrected = cur.fetchone()["corrected"]
    conn.close()
    return {
        "total": total,
        "predicted": predicted,
        "conflicts": conflicts,
        "corrected": corrected,
    }


def update_batch_item_correction(
    batch_id: str,
    item_id: int,
    corrected_label: str,
    correction_reason: str,
    correction_id: Optional[int] = None,
    previous_label: Optional[str] = None,
    previous_reason: Optional[str] = None,
    previous_operator: Optional[str] = None,
    previous_corrected_at: Optional[str] = None,
    corrected_by: Optional[str] = None,
) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE batch_items
        SET corrected_label = ?, correction_reason = ?,
            correction_id = ?, corrected_at = ?,
            previous_label = ?, previous_reason = ?,
            previous_operator = ?, previous_corrected_at = ?,
            corrected_by = ?
        WHERE id = ? AND batch_id = ?
        """,
        (
            corrected_label,
            correction_reason,
            correction_id,
            now_iso(),
            previous_label,
            previous_reason,
            previous_operator,
            previous_corrected_at,
            corrected_by,
            item_id,
            batch_id,
        ),
    )
    updated = cur.rowcount > 0
    if updated:
        cur.execute(
            """
            UPDATE batch_predictions
            SET corrected_count = (
                SELECT COUNT(*) FROM batch_items
                WHERE batch_id = ? AND corrected_label IS NOT NULL
            )
            WHERE batch_id = ?
            """,
            (batch_id, batch_id),
        )
    conn.commit()
    conn.close()
    return updated


def revert_batch_item_correction(
    batch_id: str,
    item_id: int,
    reverted_label: str,
    reverted_reason: str,
) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT previous_label, previous_reason, previous_operator, previous_corrected_at "
        "FROM batch_items WHERE id = ? AND batch_id = ?",
        (item_id, batch_id),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return False

    prev_label = row["previous_label"]
    prev_reason = row["previous_reason"]
    prev_op = row["previous_operator"]
    prev_at = row["previous_corrected_at"]

    if prev_label:
        cur.execute(
            """
            UPDATE batch_items
            SET corrected_label = ?, correction_reason = ?,
                corrected_by = ?, corrected_at = ?,
                previous_label = NULL, previous_reason = NULL,
                previous_operator = NULL, previous_corrected_at = NULL
            WHERE id = ? AND batch_id = ?
            """,
            (prev_label, prev_reason, prev_op, prev_at, item_id, batch_id),
        )
    else:
        cur.execute(
            """
            UPDATE batch_items
            SET corrected_label = NULL, correction_reason = NULL,
                correction_id = NULL, corrected_at = NULL, corrected_by = NULL,
                previous_label = NULL, previous_reason = NULL,
                previous_operator = NULL, previous_corrected_at = NULL
            WHERE id = ? AND batch_id = ?
            """,
            (item_id, batch_id),
        )

    updated = cur.rowcount > 0
    if updated:
        cur.execute(
            """
            UPDATE batch_predictions
            SET corrected_count = (
                SELECT COUNT(*) FROM batch_items
                WHERE batch_id = ? AND corrected_label IS NOT NULL
            )
            WHERE batch_id = ?
            """,
            (batch_id, batch_id),
        )
    conn.commit()
    conn.close()
    return updated


def get_batch_item(batch_id: str, item_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM batch_items WHERE batch_id = ? AND id = ?",
        (batch_id, item_id),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    r = dict_factory(row)
    r["top_predictions"] = parse_json_field(r["top_predictions"])
    r["is_conflict"] = bool(r["is_conflict"])
    return r


def export_batch_to_csv(
    batch_id: str,
    export_type: str = "prediction",
    note: Optional[str] = None,
) -> Dict[str, Any]:
    batch = get_batch(batch_id)
    if batch is None:
        raise BatchError(f"批次不存在: {batch_id}")

    items = get_batch_items(batch_id, include_conflicts=True, limit=10000)

    if export_type == "prediction":
        return _export_prediction_csv(batch, items, note)
    elif export_type == "training":
        return _export_training_csv(batch, items, note)
    else:
        raise BatchError(f"未知的导出类型: {export_type}")


def _export_prediction_csv(
    batch: Dict[str, Any],
    items: List[Dict[str, Any]],
    note: Optional[str],
) -> Dict[str, Any]:
    export_id = _gen_export_id()
    filename = f"{batch['batch_id']}_prediction.csv"
    file_path = os.path.join(EXPORTS_DIR, filename)

    rows = []
    for item in items:
        top_preds = item.get("top_predictions") or []
        top1_label = top_preds[0]["label"] if len(top_preds) > 0 else ""
        top1_prob = top_preds[0]["probability"] if len(top_preds) > 0 else 0.0
        top2_label = top_preds[1]["label"] if len(top_preds) > 1 else ""
        top2_prob = top_preds[1]["probability"] if len(top_preds) > 1 else 0.0
        top3_label = top_preds[2]["label"] if len(top_preds) > 2 else ""
        top3_prob = top_preds[2]["probability"] if len(top_preds) > 2 else 0.0

        rows.append({
            "序号": item["row_index"] + 1,
            "原文": item["clause_text"],
            "合同类型": item.get("contract_type", ""),
            "预测标签": item["predicted_label"],
            "置信度": f"{item['confidence']:.4f}",
            "Top1标签": top1_label,
            "Top1概率": f"{top1_prob:.4f}",
            "Top2标签": top2_label,
            "Top2概率": f"{top2_prob:.4f}",
            "Top3标签": top3_label,
            "Top3概率": f"{top3_prob:.4f}",
            "是否冲突": "是" if item["is_conflict"] else "否",
            "冲突原因": item.get("conflict_reason") or "",
            "真实标签": item.get("true_label") or "",
            "人工改判": item.get("corrected_label") or "",
            "改判原因": item.get("correction_reason") or "",
            "模型版本": batch["model_version"],
            "数据集版本": batch["dataset_version"],
            "批次ID": batch["batch_id"],
            "批次时间": batch["created_at"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(file_path, index=False, encoding="utf-8-sig")

    _save_export_record(
        export_id=export_id,
        batch_id=batch["batch_id"],
        export_type="prediction",
        filename=filename,
        file_path=file_path,
        model_version=batch["model_version"],
        item_count=len(items),
        note=note,
    )

    add_operation_log(
        operation_type="batch_export_prediction",
        entity_type="batch",
        entity_id=batch["batch_id"],
        details={
            "export_id": export_id,
            "export_type": "prediction",
            "item_count": len(items),
        },
    )

    return {
        "export_id": export_id,
        "export_type": "prediction",
        "filename": filename,
        "file_path": file_path,
        "item_count": len(items),
        "batch_id": batch["batch_id"],
        "model_version": batch["model_version"],
    }


def _export_training_csv(
    batch: Dict[str, Any],
    items: List[Dict[str, Any]],
    note: Optional[str],
) -> Dict[str, Any]:
    export_id = _gen_export_id()
    filename = f"{batch['batch_id']}_training.csv"
    file_path = os.path.join(EXPORTS_DIR, filename)

    corrected_items = [
        item for item in items
        if item.get("corrected_label") and item.get("corrected_label").strip()
    ]

    rows = []
    for item in corrected_items:
        rows.append({
            "条款文本": item["clause_text"],
            "风险标签": item["corrected_label"],
            "合同类型": item.get("contract_type") or "",
            "原预测标签": item["predicted_label"],
            "改判原因": item.get("correction_reason") or "",
            "来源批次": batch["batch_id"],
            "模型版本": batch["model_version"],
            "批次时间": batch["created_at"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(file_path, index=False, encoding="utf-8-sig")

    _save_export_record(
        export_id=export_id,
        batch_id=batch["batch_id"],
        export_type="training",
        filename=filename,
        file_path=file_path,
        model_version=batch["model_version"],
        item_count=len(corrected_items),
        note=note,
    )

    add_operation_log(
        operation_type="batch_export_training",
        entity_type="batch",
        entity_id=batch["batch_id"],
        details={
            "export_id": export_id,
            "export_type": "training",
            "item_count": len(corrected_items),
        },
    )

    return {
        "export_id": export_id,
        "export_type": "training",
        "filename": filename,
        "file_path": file_path,
        "item_count": len(corrected_items),
        "batch_id": batch["batch_id"],
        "model_version": batch["model_version"],
    }


def _save_export_record(
    export_id: str,
    batch_id: Optional[str],
    export_type: str,
    filename: str,
    file_path: str,
    model_version: Optional[str],
    item_count: int,
    note: Optional[str],
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO export_files
        (export_id, batch_id, export_type, filename, file_path,
         model_version, item_count, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            export_id,
            batch_id,
            export_type,
            filename,
            file_path,
            model_version,
            item_count,
            note,
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def list_exports(
    batch_id: Optional[str] = None,
    export_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM export_files WHERE 1=1"
    params: List[Any] = []
    if batch_id:
        query += " AND batch_id = ?"
        params.append(batch_id)
    if export_type:
        query += " AND export_type = ?"
        params.append(export_type)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_export(export_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM export_files WHERE export_id = ?", (export_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return dict_factory(row)


def get_batch_by_item_id(item_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bp.* FROM batch_predictions bp
        JOIN batch_items bi ON bp.batch_id = bi.batch_id
        WHERE bi.id = ?
        """,
        (item_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return dict_factory(row)


def export_audit_to_csv(
    batch_id: str,
    note: Optional[str] = None,
    role: Optional[str] = "admin",
) -> Dict[str, Any]:
    if role and not check_permission(role, "batch_audit_export"):
        raise BatchError(f"角色 '{role}' 没有审计导出的权限")

    batch = get_batch(batch_id)
    if batch is None:
        raise BatchError(f"批次不存在: {batch_id}")

    items = get_batch_items(batch_id, include_conflicts=True, limit=10000)
    all_logs = list_operation_logs(entity_type="batch", entity_id=batch_id, limit=1000)
    corr_logs = list_operation_logs(entity_type="correction", limit=5000)
    batch_corr_logs = [
        l for l in corr_logs
        if (l.get("details") or {}).get("batch_id") == batch_id
    ]
    all_operation_logs = sorted(
        all_logs + batch_corr_logs,
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )

    correction_hist = list_correction_history(batch_id=batch_id, limit=10000)

    export_id = _gen_export_id()
    filename = f"{batch['batch_id']}_audit.csv"
    file_path = os.path.join(EXPORTS_DIR, filename)

    rows = []
    for item in items:
        previous_label = item.get("previous_label") or ""
        previous_operator = item.get("previous_operator") or ""
        previous_corrected_at = item.get("previous_corrected_at") or ""

        conflict_info = ""
        if item.get("is_conflict"):
            conflict_info = item.get("conflict_reason") or "重复条款"

        recent_ops = [
            l for l in all_operation_logs
            if (l.get("details") or {}).get("batch_item_id") == item["id"]
        ][:5]
        recent_ops_str = "; ".join([
            f"{l.get('operation_type')}@{l.get('created_at','')[:19]}"
            for l in recent_ops
        ]) if recent_ops else ""

        item_corr_hist = [
            h for h in correction_hist
            if h.get("batch_item_id") == item["id"]
        ]
        change_history = "; ".join([
            f"{h.get('operation_type')}:{h.get('previous_label','')}→{h.get('new_label')}@{h.get('created_at','')[:19]}"
            for h in item_corr_hist
        ]) if item_corr_hist else ""

        rows.append({
            "序号": item["row_index"] + 1,
            "原文": item["clause_text"],
            "合同类型": item.get("contract_type", ""),
            "原始导入来源": batch.get("source_type", ""),
            "预测标签": item["predicted_label"],
            "置信度": f"{item['confidence']:.4f}" if item["confidence"] else "",
            "改判前标签": previous_label,
            "改判后标签": item.get("corrected_label") or "",
            "当前生效标签": item.get("corrected_label") or item["predicted_label"],
            "改判原因": item.get("correction_reason") or "",
            "冲突摘要": conflict_info,
            "是否冲突": "是" if item.get("is_conflict") else "否",
            "最近操作日志": recent_ops_str,
            "改判历史": change_history,
            "前次操作人": previous_operator,
            "前次改判时间": previous_corrected_at,
            "最后操作人": item.get("corrected_by") or "",
            "最后改判时间": item.get("corrected_at") or "",
            "模型版本": batch["model_version"],
            "数据集版本": batch["dataset_version"],
            "批次ID": batch["batch_id"],
            "批次时间": batch["created_at"],
            "导出时间": now_iso(),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(file_path, index=False, encoding="utf-8-sig")

    _save_export_record(
        export_id=export_id,
        batch_id=batch["batch_id"],
        export_type="audit",
        filename=filename,
        file_path=file_path,
        model_version=batch["model_version"],
        item_count=len(items),
        note=note,
    )

    add_operation_log(
        operation_type="batch_export_audit",
        entity_type="batch",
        entity_id=batch["batch_id"],
        details={
            "export_id": export_id,
            "export_type": "audit",
            "item_count": len(items),
            "role": role,
        },
    )

    return {
        "export_id": export_id,
        "export_type": "audit",
        "filename": filename,
        "file_path": file_path,
        "item_count": len(items),
        "batch_id": batch["batch_id"],
        "model_version": batch["model_version"],
    }

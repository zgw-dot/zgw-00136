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
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO batch_predictions
        (batch_id, filename, model_version, dataset_version,
         total_rows, predicted_count, conflict_count, status, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    limit: int = 50,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM batch_predictions WHERE 1=1"
    params: List[Any] = []
    if model_version:
        query += " AND model_version = ?"
        params.append(model_version)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    return rows


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
) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE batch_items
        SET corrected_label = ?, correction_reason = ?,
            correction_id = ?, corrected_at = ?
        WHERE id = ? AND batch_id = ?
        """,
        (
            corrected_label,
            correction_reason,
            correction_id,
            now_iso(),
            item_id,
            batch_id,
        ),
    )
    updated = cur.rowcount > 0
    conn.commit()
    conn.close()
    return updated


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

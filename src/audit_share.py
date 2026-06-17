import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd

from .models import get_connection, now_iso, dict_factory, parse_json_field
from .version_manager import add_operation_log, check_permission
from .batch_manager import (
    get_batch,
    get_batch_items,
    get_batch_item_count,
    list_operation_logs,
    list_correction_history,
    BatchError,
)


EXPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "exports",
)
os.makedirs(EXPORTS_DIR, exist_ok=True)


class AuditShareError(Exception):
    pass


def _gen_token() -> str:
    return f"at_{uuid.uuid4().hex}"


def _gen_export_id() -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"audit_{ts}_{uuid.uuid4().hex[:6]}"


def _add_share_log(
    token_id: Optional[int],
    batch_id: str,
    event_type: str,
    operator: Optional[str] = None,
    applicant: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_share_logs
        (token_id, batch_id, event_type, operator, applicant, details, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            token_id,
            batch_id,
            event_type,
            operator,
            applicant,
            json.dumps(details, ensure_ascii=False) if details else None,
            now_iso(),
        ),
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return new_id


def apply_for_audit_token(
    batch_id: str,
    applicant: str,
    applicant_role: str = "viewer",
    expire_hours: int = 24,
    operator: Optional[str] = None,
) -> Dict[str, Any]:
    batch = get_batch(batch_id)
    if batch is None:
        raise AuditShareError(f"批次不存在: {batch_id}")

    if not check_permission(applicant_role, "batch_audit_export"):
        _add_share_log(
            token_id=None,
            batch_id=batch_id,
            event_type="reject",
            operator=operator,
            applicant=applicant,
            details={"reason": f"角色 '{applicant_role}' 没有审计导出权限"},
        )
        raise AuditShareError(f"角色 '{applicant_role}' 没有审计导出权限")

    token_str = _gen_token()
    expires_at = (datetime.now() + timedelta(hours=expire_hours)).isoformat()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_tokens
        (token, batch_id, applicant, applicant_role, status, expires_at, created_at)
        VALUES (?, ?, ?, ?, 'active', ?, ?)
        """,
        (token_str, batch_id, applicant, applicant_role, expires_at, now_iso()),
    )
    token_id = cur.lastrowid
    conn.commit()
    conn.close()

    _add_share_log(
        token_id=token_id,
        batch_id=batch_id,
        event_type="apply",
        operator=operator,
        applicant=applicant,
        details={
            "applicant_role": applicant_role,
            "expire_hours": expire_hours,
            "expires_at": expires_at,
        },
    )

    add_operation_log(
        operation_type="audit_share_apply",
        entity_type="audit_token",
        entity_id=str(token_id),
        details={
            "batch_id": batch_id,
            "applicant": applicant,
            "applicant_role": applicant_role,
            "expires_at": expires_at,
        },
        operator=operator,
    )

    return {
        "token_id": token_id,
        "token": token_str,
        "batch_id": batch_id,
        "applicant": applicant,
        "applicant_role": applicant_role,
        "status": "active",
        "expires_at": expires_at,
        "created_at": now_iso(),
    }


def validate_audit_token(token: str) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM audit_tokens WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return {"valid": False, "reason": "凭证不存在"}

    record = dict_factory(row)

    if record["status"] == "revoked":
        return {"valid": False, "reason": "凭证已作废", "token_id": record["id"], "batch_id": record["batch_id"]}

    if record["status"] == "expired":
        return {"valid": False, "reason": "凭证已过期", "token_id": record["id"], "batch_id": record["batch_id"]}

    expires_at = datetime.fromisoformat(record["expires_at"])
    if datetime.now() > expires_at:
        conn = get_connection()
        cur2 = conn.cursor()
        cur2.execute(
            "UPDATE audit_tokens SET status = 'expired' WHERE id = ?",
            (record["id"],),
        )
        conn.commit()
        conn.close()
        return {"valid": False, "reason": "凭证已过期", "token_id": record["id"], "batch_id": record["batch_id"]}

    batch = get_batch(record["batch_id"])
    if batch is None:
        return {"valid": False, "reason": "关联批次不存在", "token_id": record["id"], "batch_id": record["batch_id"]}

    return {
        "valid": True,
        "token_id": record["id"],
        "batch_id": record["batch_id"],
        "applicant": record["applicant"],
        "applicant_role": record["applicant_role"],
        "expires_at": record["expires_at"],
    }


def download_audit_by_token(token: str, operator: Optional[str] = None) -> Dict[str, Any]:
    validation = validate_audit_token(token)
    if not validation["valid"]:
        _add_share_log(
            token_id=validation.get("token_id"),
            batch_id=validation.get("batch_id", ""),
            event_type="reject",
            operator=operator,
            applicant=None,
            details={"reason": validation["reason"], "token_used": token[:8] + "..."},
        )
        raise AuditShareError(f"凭证无效: {validation['reason']}")

    batch_id = validation["batch_id"]
    token_id = validation["token_id"]

    result = _build_audit_export(batch_id)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE audit_tokens SET download_count = download_count + 1, last_download_at = ? WHERE id = ?",
        (now_iso(), token_id),
    )
    conn.commit()
    conn.close()

    _add_share_log(
        token_id=token_id,
        batch_id=batch_id,
        event_type="download",
        operator=operator,
        applicant=validation.get("applicant"),
        details={
            "export_id": result["export_id"],
            "filename": result["filename"],
        },
    )

    add_operation_log(
        operation_type="audit_share_download",
        entity_type="audit_token",
        entity_id=str(token_id),
        details={
            "batch_id": batch_id,
            "export_id": result["export_id"],
            "applicant": validation.get("applicant"),
        },
        operator=operator,
    )

    return result


def _build_audit_export(batch_id: str) -> Dict[str, Any]:
    batch = get_batch(batch_id)
    if batch is None:
        raise AuditShareError(f"批次不存在: {batch_id}")

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
    filename = f"{batch['batch_id']}_audit_share.csv"
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
            "改判前标签": previous_label or item["predicted_label"],
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
            "绑定模型版本": batch["model_version"],
            "绑定数据集版本": batch["dataset_version"],
            "批次ID": batch["batch_id"],
            "批次时间": batch["created_at"],
            "导出时间": now_iso(),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(file_path, index=False, encoding="utf-8-sig")

    _save_audit_export_record(
        export_id=export_id,
        batch_id=batch["batch_id"],
        filename=filename,
        file_path=file_path,
        model_version=batch["model_version"],
        item_count=len(items),
    )

    return {
        "export_id": export_id,
        "export_type": "audit_share",
        "filename": filename,
        "file_path": file_path,
        "item_count": len(items),
        "batch_id": batch["batch_id"],
        "model_version": batch["model_version"],
    }


def _save_audit_export_record(
    export_id: str,
    batch_id: str,
    filename: str,
    file_path: str,
    model_version: str,
    item_count: int,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO export_files
        (export_id, batch_id, export_type, filename, file_path,
         model_version, item_count, note, created_at)
        VALUES (?, ?, 'audit_share', ?, ?, ?, ?, ?, ?)
        """,
        (
            export_id,
            batch_id,
            filename,
            file_path,
            model_version,
            item_count,
            "受控审计共享导出",
            now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def revoke_audit_token(
    token_id: int,
    revoked_by: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM audit_tokens WHERE id = ?", (token_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise AuditShareError(f"凭证不存在: {token_id}")

    record = dict_factory(row)
    if record["status"] != "active":
        conn.close()
        raise AuditShareError(f"凭证状态为 '{record['status']}'，无法作废")

    cur.execute(
        "UPDATE audit_tokens SET status = 'revoked', revoked_at = ?, revoked_by = ? WHERE id = ?",
        (now_iso(), revoked_by, token_id),
    )
    conn.commit()
    conn.close()

    _add_share_log(
        token_id=token_id,
        batch_id=record["batch_id"],
        event_type="revoke",
        operator=revoked_by,
        applicant=record["applicant"],
        details={
            "original_applicant_role": record["applicant_role"],
        },
    )

    add_operation_log(
        operation_type="audit_share_revoke",
        entity_type="audit_token",
        entity_id=str(token_id),
        details={
            "batch_id": record["batch_id"],
            "applicant": record["applicant"],
            "revoked_by": revoked_by,
        },
        operator=revoked_by,
    )

    return {
        "token_id": token_id,
        "status": "revoked",
        "revoked_at": now_iso(),
        "revoked_by": revoked_by,
    }


def reissue_audit_token(
    token_id: int,
    applicant: Optional[str] = None,
    applicant_role: Optional[str] = None,
    expire_hours: int = 24,
    operator: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM audit_tokens WHERE id = ?", (token_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise AuditShareError(f"原凭证不存在: {token_id}")

    old_record = dict_factory(row)

    if old_record["status"] == "active":
        cur.execute(
            "UPDATE audit_tokens SET status = 'revoked', revoked_at = ?, revoked_by = ? WHERE id = ?",
            (now_iso(), operator or "system_reissue", token_id),
        )
        conn.commit()

    conn.close()

    actual_applicant = applicant or old_record["applicant"]
    actual_role = applicant_role or old_record["applicant_role"]
    batch_id = old_record["batch_id"]

    new_token_str = _gen_token()
    expires_at = (datetime.now() + timedelta(hours=expire_hours)).isoformat()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_tokens
        (token, batch_id, applicant, applicant_role, status, expires_at, created_at, reissued_from)
        VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
        """,
        (new_token_str, batch_id, actual_applicant, actual_role, expires_at, now_iso(), token_id),
    )
    new_token_id = cur.lastrowid
    conn.commit()
    conn.close()

    _add_share_log(
        token_id=new_token_id,
        batch_id=batch_id,
        event_type="reissue",
        operator=operator,
        applicant=actual_applicant,
        details={
            "old_token_id": token_id,
            "old_token_status": old_record["status"],
            "new_expires_at": expires_at,
        },
    )

    add_operation_log(
        operation_type="audit_share_reissue",
        entity_type="audit_token",
        entity_id=str(new_token_id),
        details={
            "old_token_id": token_id,
            "batch_id": batch_id,
            "applicant": actual_applicant,
            "reissued_by": operator,
        },
        operator=operator,
    )

    return {
        "token_id": new_token_id,
        "token": new_token_str,
        "batch_id": batch_id,
        "applicant": actual_applicant,
        "applicant_role": actual_role,
        "status": "active",
        "expires_at": expires_at,
        "reissued_from": token_id,
        "created_at": now_iso(),
    }


def list_audit_tokens(
    batch_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM audit_tokens WHERE 1=1"
    params: List[Any] = []
    if batch_id:
        query += " AND batch_id = ?"
        params.append(batch_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_audit_token(token_id: Optional[int] = None, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    if token_id:
        cur.execute("SELECT * FROM audit_tokens WHERE id = ?", (token_id,))
    elif token:
        cur.execute("SELECT * FROM audit_tokens WHERE token = ?", (token,))
    else:
        conn.close()
        return None
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return dict_factory(row)


def list_audit_share_logs(
    batch_id: Optional[str] = None,
    token_id: Optional[int] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM audit_share_logs WHERE 1=1"
    params: List[Any] = []
    if batch_id:
        query += " AND batch_id = ?"
        params.append(batch_id)
    if token_id:
        query += " AND token_id = ?"
        params.append(token_id)
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict_factory(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r["details"] = parse_json_field(r["details"])
    return rows


def expire_audit_tokens() -> int:
    conn = get_connection()
    cur = conn.cursor()
    now = now_iso()
    cur.execute(
        "UPDATE audit_tokens SET status = 'expired' WHERE status = 'active' AND expires_at < ?",
        (now,),
    )
    count = cur.rowcount
    conn.commit()
    conn.close()
    return count

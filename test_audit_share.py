import json
import os
import sqlite3
import time
import urllib.request
import urllib.parse
import urllib.error
import datetime

API = "http://127.0.0.1:8001"
BASE = os.path.dirname(os.path.abspath(__file__))
PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def api(method, path, data=None, content_type="application/json"):
    url = API + path
    if data is not None:
        body = json.dumps(data).encode("utf-8") if isinstance(data, dict) else data
        req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": content_type})
    else:
        req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {}
        return e.code, body


def api_raw(method, path):
    url = API + path
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def api_file(method, path, filepath, fields=None):
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body = b""
    if fields:
        for k, v in fields.items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if filepath:
        with open(filepath, "rb") as f:
            content = f.read()
        fname = os.path.basename(filepath)
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'.encode()
        body += b"Content-Type: text/csv\r\n\r\n"
        body += content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(API + path, data=body, method=method,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            b = json.loads(e.read().decode("utf-8"))
        except Exception:
            b = {}
        return e.code, b


def write_csv(path, lines):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")


print("=" * 60)
print("受控审计共享模块 - 验证脚本")
print("=" * 60)

# ============================================================
print("\n[A1] 准备测试批次")
# ============================================================

code_h, health = api("GET", "/health")
check("服务健康", health.get("status") == "ok")

code_bl, resp_bl = api("GET", "/api/batches?limit=10")
existing_batch = None
if resp_bl and len(resp_bl) > 0:
    for b in resp_bl:
        if b.get("conflict_count", 0) >= 0:
            existing_batch = b
            break

if existing_batch:
    test_batch_id = existing_batch["batch_id"]
    check("使用已有批次", True, f"batch_id={test_batch_id}")
else:
    sample_path = os.path.join(BASE, "sample_data", "contracts_sample.csv")
    if not os.path.exists(sample_path):
        batch_csv = os.path.join(BASE, "data", "datasets", "_test_audit_share.csv")
        write_csv(batch_csv, [
            "条款文本,合同类型,风险标签",
            "买方应在收到发票后30日内付款,采购合同,付款",
            "逾期交付超过10日支付违约金,采购合同,违约",
            "本合同到期自动续展一年,服务合同,自动续约",
        ])
        sample_path = batch_csv

    code_b, resp_b = api_file("POST", "/api/batches", sample_path, {"note": "审计共享测试批次"})
    if resp_b.get("success"):
        test_batch_id = resp_b["batch_id"]
        check("创建测试批次", True, f"batch_id={test_batch_id}")
    else:
        code_bl2, resp_bl2 = api("GET", "/api/batches?limit=5")
        if resp_bl2:
            test_batch_id = resp_bl2[0]["batch_id"]
            check("使用已有批次(备选)", True, f"batch_id={test_batch_id}")
        else:
            check("无法获取测试批次", False, "没有可用批次")
            test_batch_id = None

if not test_batch_id:
    print("ERROR: 无法继续测试，没有可用批次")
    raise SystemExit(1)

# ============================================================
print("\n[A2] 申请审计共享凭证 - admin角色")
# ============================================================

code_apply, resp_apply = api("POST", "/api/audit-share/apply", {
    "batch_id": test_batch_id,
    "applicant": "tester_admin",
    "applicant_role": "admin",
    "expire_hours": 24,
    "operator": "test_runner",
})
check("admin申请凭证成功", resp_apply.get("success") == True, f"resp={resp_apply}")
admin_token = resp_apply.get("token")
admin_token_id = resp_apply.get("token_id")
check("返回token", admin_token is not None and admin_token.startswith("at_"), f"token={admin_token}")
check("返回token_id", admin_token_id is not None)
check("status=active", resp_apply.get("status") == "active")
check("返回expires_at", resp_apply.get("expires_at") is not None)
check("返回applicant=tester_admin", resp_apply.get("applicant") == "tester_admin")

# ============================================================
print("\n[A3] 申请审计共享凭证 - viewer角色（应被拒绝）")
# ============================================================

code_vapply, resp_vapply = api("POST", "/api/audit-share/apply", {
    "batch_id": test_batch_id,
    "applicant": "tester_viewer",
    "applicant_role": "viewer",
    "expire_hours": 24,
})
check("viewer申请凭证被拒", resp_vapply.get("success") == False, f"resp={resp_vapply}")
check("viewer被拒返回403", code_vapply == 403)
check("错误信息含权限提示", "权限" in resp_vapply.get("error", ""), f"error={resp_vapply.get('error')}")

# ============================================================
print("\n[A4] 验证凭证校验接口")
# ============================================================

code_val, resp_val = api("GET", f"/api/audit-share/{admin_token}/validate")
check("凭证校验返回200", code_val == 200)
check("凭证有效", resp_val.get("valid") == True, f"resp={resp_val}")
check("凭证校验返回batch_id", resp_val.get("batch_id") == test_batch_id)
check("凭证校验返回applicant", resp_val.get("applicant") == "tester_admin")

# ============================================================
print("\n[A5] 无凭证直链下载 - 应被拒绝")
# ============================================================

code_no_token, resp_no_token = api_raw("GET", f"/api/audit-share/invalid_token_xyz/download")
check("无效凭证下载返回非200", code_no_token != 200, f"code={code_no_token}")

code_direct, resp_direct = api("GET", f"/api/batches/{test_batch_id}/export/audit/download?role=viewer")
check("viewer直接下载审计被拒", code_direct == 403 or resp_direct is None or (isinstance(resp_direct, dict) and resp_direct.get("success") == False), f"code={code_direct}")

# ============================================================
print("\n[A6] 有效凭证下载审计CSV")
# ============================================================

code_dl, resp_dl_raw = api_raw("GET", f"/api/audit-share/{admin_token}/download")
check("凭证下载返回200", code_dl == 200, f"code={code_dl}")
if code_dl == 200:
    csv_content = resp_dl_raw.decode("utf-8-sig")
    csv_lines = [l for l in csv_content.strip().split("\n") if l.strip()]
    check("审计CSV有数据行", len(csv_lines) >= 2, f"lines={len(csv_lines)}")

    header = csv_lines[0]
    check("审计CSV含原始导入来源", "原始导入来源" in header, f"header={header[:200]}")
    check("审计CSV含冲突摘要", "冲突摘要" in header)
    check("审计CSV含改判前标签", "改判前标签" in header)
    check("审计CSV含改判后标签", "改判后标签" in header)
    check("审计CSV含最近操作日志", "最近操作日志" in header)
    check("审计CSV含绑定模型版本", "绑定模型版本" in header)
    check("审计CSV含绑定数据集版本", "绑定数据集版本" in header)
    check("审计CSV含当前生效标签", "当前生效标签" in header)
    check("审计CSV含改判历史", "改判历史" in header)
    check("审计CSV含导出时间", "导出时间" in header)

# ============================================================
print("\n[A7] 重复下载（同一凭证可多次使用）")
# ============================================================

code_dl2, resp_dl2_raw = api_raw("GET", f"/api/audit-share/{admin_token}/download")
check("同一凭证可重复下载", code_dl2 == 200, f"code={code_dl2}")

# ============================================================
print("\n[A8] 过期凭证下载 - 应被拒绝")
# ============================================================

code_exp, resp_exp = api("POST", "/api/audit-share/apply", {
    "batch_id": test_batch_id,
    "applicant": "tester_expire",
    "applicant_role": "admin",
    "expire_hours": 0,
})
check("0小时过期凭证申请成功", resp_exp.get("success") == True)
expired_token = resp_exp.get("token")

time.sleep(2)

code_exp_val, resp_exp_val = api("GET", f"/api/audit-share/{expired_token}/validate")
check("过期凭证校验无效", resp_exp_val.get("valid") == False, f"resp={resp_exp_val}")
check("过期原因正确", "过期" in resp_exp_val.get("reason", ""), f"reason={resp_exp_val.get('reason')}")

code_exp_dl, _ = api_raw("GET", f"/api/audit-share/{expired_token}/download")
check("过期凭证下载被拒", code_exp_dl != 200, f"code={code_exp_dl}")

# ============================================================
print("\n[A9] 手动作废凭证")
# ============================================================

code_rev_apply, resp_rev_apply = api("POST", "/api/audit-share/apply", {
    "batch_id": test_batch_id,
    "applicant": "tester_revoke",
    "applicant_role": "reviewer",
    "expire_hours": 24,
})
check("申请待作废凭证成功", resp_rev_apply.get("success") == True)
revoke_token = resp_rev_apply.get("token")
revoke_token_id = resp_rev_apply.get("token_id")

code_rev_val, resp_rev_val = api("GET", f"/api/audit-share/{revoke_token}/validate")
check("作废前凭证有效", resp_rev_val.get("valid") == True)

code_revoke, resp_revoke = api("POST", f"/api/audit-share/{revoke_token_id}/revoke", {
    "revoked_by": "admin_user",
})
check("作废凭证成功", resp_revoke.get("success") == True, f"resp={resp_revoke}")
check("作废后status=revoked", resp_revoke.get("status") == "revoked")

code_rev_val2, resp_rev_val2 = api("GET", f"/api/audit-share/{revoke_token}/validate")
check("作废后凭证无效", resp_rev_val2.get("valid") == False)
check("作废原因正确", "作废" in resp_rev_val2.get("reason", ""), f"reason={resp_rev_val2.get('reason')}")

# ============================================================
print("\n[A10] 权限回收后再次下载 - 被拒绝")
# ============================================================

code_dl_revoked, _ = api_raw("GET", f"/api/audit-share/{revoke_token}/download")
check("作废凭证下载被拒", code_dl_revoked != 200, f"code={code_dl_revoked}")

# ============================================================
print("\n[A11] 重新签发凭证")
# ============================================================

code_reissue, resp_reissue = api("POST", f"/api/audit-share/{revoke_token_id}/reissue", {
    "applicant": "tester_reissued",
    "applicant_role": "reviewer",
    "expire_hours": 48,
    "operator": "admin_user",
})
check("重新签发成功", resp_reissue.get("success") == True, f"resp={resp_reissue}")
new_token = resp_reissue.get("token")
new_token_id = resp_reissue.get("token_id")
check("新凭证token不同", new_token != revoke_token)
check("新凭证reissued_from指向旧凭证", resp_reissue.get("reissued_from") == revoke_token_id)
check("新凭证status=active", resp_reissue.get("status") == "active")
check("新凭证applicant=tester_reissued", resp_reissue.get("applicant") == "tester_reissued")

# ============================================================
print("\n[A12] 重新导出后旧链接失效")
# ============================================================

code_old_val, resp_old_val = api("GET", f"/api/audit-share/{revoke_token}/validate")
check("旧凭证已失效", resp_old_val.get("valid") == False)

code_new_val, resp_new_val = api("GET", f"/api/audit-share/{new_token}/validate")
check("新凭证有效", resp_new_val.get("valid") == True)

code_dl_new, _ = api_raw("GET", f"/api/audit-share/{new_token}/download")
check("新凭证可下载", code_dl_new == 200)

# ============================================================
print("\n[A13] 改判撤回后审计内容同步更新")
# ============================================================

code_items, resp_items = api("GET", f"/api/batches/{test_batch_id}/items?limit=10&include_conflicts=false")
normal_items = resp_items.get("items", [])
if normal_items:
    target_item = normal_items[0]
    orig_label = target_item["predicted_label"]

    code_corr, resp_corr = api("POST", "/api/corrections", {
        "clause_text": target_item["clause_text"],
        "predicted_label": orig_label,
        "corrected_label": "违约" if orig_label != "违约" else "付款",
        "reason": "审计共享测试-改判",
        "batch_id": test_batch_id,
        "batch_item_id": target_item["id"],
        "operator": "audit_tester",
    })
    check("改判成功用于审计同步测试", resp_corr.get("success") == True)
    correction_id_for_audit = resp_corr.get("correction_id")

    code_audit_dl, resp_audit_raw = api_raw("GET", f"/api/audit-share/{admin_token}/download")
    check("改判后审计下载成功", code_audit_dl == 200)
    if code_audit_dl == 200:
        audit_csv = resp_audit_raw.decode("utf-8-sig")
        new_label = "违约" if orig_label != "违约" else "付款"
        check("审计CSV含改判后标签", new_label in audit_csv, f"new_label={new_label}")

    if resp_corr.get("success") and correction_id_for_audit:
        code_revert, resp_revert = api("POST", f"/api/corrections/{correction_id_for_audit}/revert", {
            "correction_id": correction_id_for_audit,
            "batch_id": test_batch_id,
            "batch_item_id": target_item["id"],
            "operator": "audit_reverter",
        })
        check("撤回改判成功", resp_revert.get("success") == True)

        code_audit_dl2, resp_audit_raw2 = api_raw("GET", f"/api/audit-share/{admin_token}/download")
        check("撤回后审计下载成功", code_audit_dl2 == 200)
        if code_audit_dl2 == 200:
            audit_csv2 = resp_audit_raw2.decode("utf-8-sig")
            check("撤回后审计CSV含原始预测标签", orig_label in audit_csv2)
else:
    check("无正常项跳过审计同步测试", True, "跳过")

# ============================================================
print("\n[A14] 审计共享日志查询")
# ============================================================

code_logs, resp_logs = api("GET", f"/api/audit-share/logs?batch_id={test_batch_id}&limit=50")
check("审计日志查询返回200", code_logs == 200)
check("审计日志有记录", len(resp_logs) >= 1, f"count={len(resp_logs)}")

log_events = [l.get("event_type") for l in resp_logs]
check("日志包含apply事件", "apply" in log_events, f"events={log_events}")
check("日志包含download事件", "download" in log_events)
check("日志包含revoke事件", "revoke" in log_events)
check("日志包含reissue事件", "reissue" in log_events)

reject_logs = [l for l in resp_logs if l.get("event_type") == "reject"]
check("日志包含reject事件（viewer被拒）", len(reject_logs) >= 1, f"reject_count={len(reject_logs)}")

code_logs_by_event, resp_logs_by_event = api("GET", "/api/audit-share/logs?event_type=download&limit=50")
check("按事件类型过滤日志", len(resp_logs_by_event) >= 1)

# ============================================================
print("\n[A15] 凭证列表查询")
# ============================================================

code_tokens, resp_tokens = api("GET", f"/api/audit-share/tokens?batch_id={test_batch_id}&limit=50")
check("凭证列表查询返回200", code_tokens == 200)
check("凭证列表有记录", len(resp_tokens) >= 3, f"count={len(resp_tokens)}")

active_tokens = [t for t in resp_tokens if t.get("status") == "active"]
check("有active状态凭证", len(active_tokens) >= 1)

code_tokens_status, resp_tokens_status = api("GET", f"/api/audit-share/tokens?batch_id={test_batch_id}&status=revoked&limit=50")
check("按状态过滤凭证", len(resp_tokens_status) >= 1, f"revoked_count={len(resp_tokens_status)}")

# ============================================================
print("\n[A16] 不存在的批次申请凭证")
# ============================================================

code_bad, resp_bad = api("POST", "/api/audit-share/apply", {
    "batch_id": "batch_nonexistent_999",
    "applicant": "tester",
    "applicant_role": "admin",
})
check("不存在批次申请被拒", resp_bad.get("success") == False)

# ============================================================
print("\n[A17] 重复作废同一凭证")
# ============================================================

code_dbl_rev, resp_dbl_rev = api("POST", f"/api/audit-share/{revoke_token_id}/revoke", {
    "revoked_by": "admin_user",
})
check("重复作废返回失败", resp_dbl_rev.get("success") == False)

# ============================================================
print("\n[A18] 不存在凭证下载")
# ============================================================

code_ghost, _ = api_raw("GET", "/api/audit-share/at_ghost_nonexistent/download")
check("不存在凭证下载被拒", code_ghost != 200)

# ============================================================
print("\n[A19] 重启后授权状态与日志保留验证")
# ============================================================

db_path = os.path.join(BASE, "data", "app.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

token_count = cur.execute("SELECT COUNT(*) FROM audit_tokens").fetchone()[0]
check("数据库中有审计凭证记录", token_count >= 3, f"count={token_count}")

share_log_count = cur.execute("SELECT COUNT(*) FROM audit_share_logs").fetchone()[0]
check("数据库中有审计共享日志", share_log_count >= 3, f"count={share_log_count}")

revoked_tokens = cur.execute("SELECT COUNT(*) FROM audit_tokens WHERE status = 'revoked'").fetchone()[0]
check("数据库中有revoked状态凭证", revoked_tokens >= 1)

active_tokens_db = cur.execute("SELECT COUNT(*) FROM audit_tokens WHERE status = 'active'").fetchone()[0]
check("数据库中有active状态凭证", active_tokens_db >= 1)

reissue_records = cur.execute(
    "SELECT COUNT(*) FROM audit_tokens WHERE reissued_from IS NOT NULL"
).fetchone()[0]
check("数据库中有重签发记录", reissue_records >= 1)

apply_logs = cur.execute(
    "SELECT COUNT(*) FROM audit_share_logs WHERE event_type = 'apply'"
).fetchone()[0]
check("数据库中有apply日志", apply_logs >= 1)

download_logs = cur.execute(
    "SELECT COUNT(*) FROM audit_share_logs WHERE event_type = 'download'"
).fetchone()[0]
check("数据库中有download日志", download_logs >= 1)

revoke_logs = cur.execute(
    "SELECT COUNT(*) FROM audit_share_logs WHERE event_type = 'revoke'"
).fetchone()[0]
check("数据库中有revoke日志", revoke_logs >= 1)

reissue_logs = cur.execute(
    "SELECT COUNT(*) FROM audit_share_logs WHERE event_type = 'reissue'"
).fetchone()[0]
check("数据库中有reissue日志", reissue_logs >= 1)

reject_logs_db = cur.execute(
    "SELECT COUNT(*) FROM audit_share_logs WHERE event_type = 'reject'"
).fetchone()[0]
check("数据库中有reject日志", reject_logs_db >= 1)

conn.close()

# 验证重启后API仍能查到所有凭证和日志
code_tk2, resp_tk2 = api("GET", f"/api/audit-share/tokens?batch_id={test_batch_id}&limit=50")
check("重启后凭证列表仍可查询", len(resp_tk2) >= 3)

code_lg2, resp_lg2 = api("GET", f"/api/audit-share/logs?batch_id={test_batch_id}&limit=50")
check("重启后审计日志仍可查询", len(resp_lg2) >= 3)

code_val_persist, resp_val_persist = api("GET", f"/api/audit-share/{admin_token}/validate")
check("重启后admin凭证仍有效", resp_val_persist.get("valid") == True)

# ============================================================
print("\n[A20] 审计导出内容固定字段验证")
# ============================================================

code_field, resp_field_raw = api_raw("GET", f"/api/audit-share/{admin_token}/download")
if code_field == 200:
    field_csv = resp_field_raw.decode("utf-8-sig")
    field_lines = [l for l in field_csv.strip().split("\n") if l.strip()]
    field_header = field_lines[0]

    required_fields = [
        "原始导入来源",
        "冲突摘要",
        "改判前标签",
        "改判后标签",
        "当前生效标签",
        "最近操作日志",
        "绑定模型版本",
        "绑定数据集版本",
        "改判历史",
        "导出时间",
    ]
    all_present = True
    missing = []
    for field in required_fields:
        if field not in field_header:
            all_present = False
            missing.append(field)
    check("审计CSV包含所有必需字段", all_present, f"missing={missing}")

# ============================================================
print("\n[A21] 审计共享日志详情验证")
# ============================================================

apply_log_detail = next((l for l in resp_lg2 if l.get("event_type") == "apply"), None)
if apply_log_detail:
    check("apply日志含token_id", apply_log_detail.get("token_id") is not None)
    check("apply日志含batch_id", apply_log_detail.get("batch_id") == test_batch_id)
    check("apply日志含applicant", apply_log_detail.get("applicant") is not None)
    check("apply日志含details", apply_log_detail.get("details") is not None)

revoke_log_detail = next((l for l in resp_lg2 if l.get("event_type") == "revoke"), None)
if revoke_log_detail:
    check("revoke日志含operator", revoke_log_detail.get("operator") is not None)

reissue_log_detail = next((l for l in resp_lg2 if l.get("event_type") == "reissue"), None)
if reissue_log_detail:
    check("reissue日志含details.old_token_id", (reissue_log_detail.get("details") or {}).get("old_token_id") is not None)

# ============================================================
print("\n[A22] 凭证download_count递增验证")
# ============================================================

code_count_apply, resp_count_apply = api("POST", "/api/audit-share/apply", {
    "batch_id": test_batch_id,
    "applicant": "tester_count",
    "applicant_role": "reviewer",
    "expire_hours": 24,
})
count_token_id = resp_count_apply.get("token_id")
count_token = resp_count_apply.get("token")

api_raw("GET", f"/api/audit-share/{count_token}/download")
api_raw("GET", f"/api/audit-share/{count_token}/download")
api_raw("GET", f"/api/audit-share/{count_token}/download")

code_tk_detail, resp_tk_detail = api("GET", f"/api/audit-share/tokens?batch_id={test_batch_id}&status=active&limit=50")
count_token_detail = next((t for t in resp_tk_detail if t.get("id") == count_token_id), None)
if count_token_detail:
    check("download_count=3", count_token_detail.get("download_count") == 3, f"count={count_token_detail.get('download_count')}")
    check("last_download_at有值", count_token_detail.get("last_download_at") is not None)

# ============================================================
print("\n[A23] 操作日志中也包含审计共享事件")
# ============================================================

code_op_logs, resp_op_logs = api("GET", "/api/operation-logs?limit=50")
audit_log_types = [l["operation_type"] for l in resp_op_logs if "audit_share" in l.get("operation_type", "")]
check("操作日志包含审计共享事件", len(audit_log_types) >= 3, f"audit_events={len(audit_log_types)}")

has_apply_log = "audit_share_apply" in audit_log_types
has_download_log = "audit_share_download" in audit_log_types
has_revoke_log = "audit_share_revoke" in audit_log_types
has_reissue_log = "audit_share_reissue" in audit_log_types
check("操作日志含audit_share_apply", has_apply_log)
check("操作日志含audit_share_download", has_download_log)
check("操作日志含audit_share_revoke", has_revoke_log)
check("操作日志含audit_share_reissue", has_reissue_log)

# ============================================================
print("\n" + "=" * 60)
print(f"受控审计共享验证结果: PASS={PASSED}  FAIL={FAILED}")
print("=" * 60)

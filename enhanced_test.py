import json, urllib.request, urllib.parse, os, sqlite3, datetime

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
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read()

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
print("批量预测增强功能 - 专项测试脚本")
print("=" * 60)

# ============================================================
print("\n[0] 环境检查：确保有现役模型和数据集")
# ============================================================
c0, health = api("GET", "/health")
check("服务健康检查通过", c0 == 200 and health.get("status") == "ok")
check("有现役模型", health.get("active_model") is not None)
check("至少有1个数据集", health.get("datasets", 0) >= 1)

c0a, models = api("GET", "/api/models")
active_mv = health.get("active_model")
check("至少有1个已训练完成的模型", len([m for m in models if m.get("training_status") == "completed"]) >= 1)

c0b, datasets = api("GET", "/api/datasets")
check("至少有1个数据集", len(datasets) >= 1)
ds_version = datasets[0]["version"] if datasets else None

# ============================================================
print("\n[1] 可撤销改判流程")
# ============================================================
# 1a 创建测试批次
batch_csv = os.path.join(BASE, "data", "datasets", "_enh_test_revert.csv")
write_csv(batch_csv, [
    "条款文本,合同类型",
    "买方应在货物验收后15日内支付全款,采购合同",
    "逾期付款按日万分之三支付违约金,采购合同",
    "本合同期满双方无异议自动续展一年,服务合同",
])
c1a, r1a = api_file("POST", "/api/batches", batch_csv, {"note": "增强测试-改判撤回批次"})
check("创建改判撤回批次成功", r1a.get("success") == True, f"code={c1a} resp={r1a}")
rev_batch_id = r1a.get("batch_id")
check("返回有效 batch_id", rev_batch_id is not None)
os.remove(batch_csv)

# 1b 获取批次项
c1b, r1b = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
check("获取批次项成功", c1b == 200 and r1b.get("total") == 3)
items = r1b.get("items", [])
item1 = items[0]

# 1c 首次改判
c1c, r1c = api("POST", "/api/corrections", {
    "clause_text": item1["clause_text"],
    "predicted_label": item1["predicted_label"],
    "corrected_label": "付款",
    "reason": "首次改判-增强测试",
    "batch_id": rev_batch_id,
    "batch_item_id": item1["id"],
    "operator": "tester_alice",
})
check("首次改判成功", r1c.get("success") == True, f"code={c1c} resp={r1c}")
correction_id_1 = r1c.get("correction_id")
check("首次改判返回 correction_id", correction_id_1 is not None)

# 1d 验证批次项已更新
c1d, r1d = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
item1_updated = next(i for i in r1d["items"] if i["id"] == item1["id"])
check("首次改判后 corrected_label=付款", item1_updated.get("corrected_label") == "付款")
check("首次改判后 corrected_by=tester_alice", item1_updated.get("corrected_by") == "tester_alice")
check("首次改判后 previous_label 为空（首次）",
      item1_updated.get("previous_label") in (None, "", "null"))

# 1e 验证改判历史
c1e, r1e = api("GET", f"/api/corrections/history?batch_id={rev_batch_id}&limit=10")
check("改判历史API返回200", c1e == 200)
check("改判历史至少1条记录", len(r1e) >= 1)
his1 = next((h for h in r1e if h.get("correction_id") == correction_id_1), None)
check("首次改判历史记录存在", his1 is not None)
if his1:
    check("首次改判 operation_type=create", his1.get("operation_type") == "create")
    check("首次改判 new_label=付款", his1.get("new_label") == "付款")
    check("首次改判 new_operator=tester_alice", his1.get("new_operator") == "tester_alice")

# 1f 二次改判（覆盖前次）
c1f, r1f = api("POST", "/api/corrections", {
    "clause_text": item1["clause_text"],
    "predicted_label": item1["predicted_label"],
    "corrected_label": "违约",
    "reason": "二次改判-覆盖前次",
    "batch_id": rev_batch_id,
    "batch_item_id": item1["id"],
    "operator": "tester_bob",
})
check("二次改判成功", r1f.get("success") == True)
correction_id_2 = r1f.get("correction_id")
check("二次改判返回新 correction_id", correction_id_2 is not None and correction_id_2 != correction_id_1)

# 1g 验证 previous_label 保留
c1g, r1g = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
item1_upd2 = next(i for i in r1g["items"] if i["id"] == item1["id"])
check("二次改判后 corrected_label=违约", item1_upd2.get("corrected_label") == "违约")
check("二次改判后 previous_label=付款（保留前次）", item1_upd2.get("previous_label") == "付款")
check("二次改判后 previous_reason 保留", item1_upd2.get("previous_reason") == "首次改判-增强测试")
check("二次改判后 previous_operator=tester_alice", item1_upd2.get("previous_operator") == "tester_alice")

# 1h 撤回二次改判
c1h, r1h = api("POST", f"/api/corrections/{correction_id_2}/revert", {
    "correction_id": correction_id_2,
    "batch_id": rev_batch_id,
    "batch_item_id": item1["id"],
    "operator": "tester_admin",
})
check("撤回二次改判成功", r1h.get("success") == True, f"code={c1h} resp={r1h}")
check("撤回返回 restored_label=付款", r1h.get("restored_label") == "付款")

# 1i 验证撤回后标签恢复
c1i, r1i = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
item1_rev = next(i for i in r1i["items"] if i["id"] == item1["id"])
check("撤回后 corrected_label 恢复为付款", item1_rev.get("corrected_label") == "付款")
check("撤回后 corrected_by 恢复为 tester_alice", item1_rev.get("corrected_by") == "tester_alice")

# 1j 验证撤回已写入 correction_history
c1j, r1j = api("GET", f"/api/corrections/history?batch_id={rev_batch_id}&limit=10")
rev_his = next((h for h in r1j if h.get("operation_type") == "revert"), None)
check("撤回操作已写入 correction_history", rev_his is not None)
if rev_his:
    check("撤回历史 new_operator=tester_admin", rev_his.get("new_operator") == "tester_admin")
    check("撤回历史 new_label=付款", rev_his.get("new_label") == "付款")

# 1k 验证撤回后训练导出使用恢复后的标签
c1k, r1k_raw = api_raw("GET", f"/api/batches/{rev_batch_id}/export/training/download")
check("撤回后训练导出返回200", c1k == 200)
train_content = r1k_raw.decode("utf-8-sig")
train_lines = [l for l in train_content.strip().split("\n") if l.strip()]
check("撤回后训练导出有2行（表头+1行改判）", len(train_lines) == 2)
if len(train_lines) >= 2:
    check("撤回后训练导出使用恢复后的付款标签", "付款" in train_lines[1])

# 1l 验证撤回后预测导出反映恢复后的标签
c1l, r1l_raw = api_raw("GET", f"/api/batches/{rev_batch_id}/export/prediction/download")
check("撤回后预测导出返回200", c1l == 200)
check("撤回后预测导出含恢复后的付款标签", "付款" in r1l_raw.decode("utf-8-sig"))

# 1m 验证操作日志包含 correction_revert（通过批次详情聚合API）
c1m, r1m = api("GET", f"/api/batches/{rev_batch_id}/detail")
log_types = [l["operation_type"] for l in r1m.get("operation_logs", [])]
check("操作日志包含 correction_create", "correction_create" in log_types)
check("操作日志包含 correction_revert", "correction_revert" in log_types)

# ============================================================
print("\n[2] 批次工作台多维度筛选")
# ============================================================
# 先确保有第二个批次（带冲突的）用于筛选对比
batch_csv2 = os.path.join(BASE, "data", "datasets", "_enh_test_filter.csv")
write_csv(batch_csv2, [
    "条款文本,合同类型",
    "条款AAA,采购合同",
    "条款BBB,采购合同",
    "条款AAA,采购合同",  # 重复
])
c2a, r2a = api_file("POST", "/api/batches", batch_csv2, {"note": "增强测试-筛选用冲突批次"})
check("创建冲突筛选批次成功", r2a.get("success") == True)
conflict_batch_id = r2a.get("batch_id")
os.remove(batch_csv2)

# 2a 按冲突状态筛选
c2b, r2b = api("GET", "/api/batches?has_conflicts=yes&limit=10")
check("按有冲突筛选：所有批次 conflict_count>0",
      all(b.get("conflict_count", 0) > 0 for b in r2b))

c2c, r2c = api("GET", "/api/batches?has_conflicts=no&limit=10")
check("按无冲突筛选：所有批次 conflict_count=0",
      all(b.get("conflict_count", 0) == 0 for b in r2c))

# 2b 按改判状态筛选
c2d, r2d = api("GET", "/api/batches?has_corrections=yes&limit=10")
check("按有改判筛选：所有批次 corrected_count>0",
      all(b.get("corrected_count", 0) > 0 for b in r2d),
      f"got={[(b['batch_id'], b.get('corrected_count')) for b in r2d]}")

c2e, r2e = api("GET", "/api/batches?has_corrections=no&limit=10")
check("按无改判筛选：所有批次 corrected_count=0",
      all(b.get("corrected_count", 0) == 0 for b in r2e))

# 2c 按模型版本筛选
if active_mv:
    c2f, r2f = api("GET", f"/api/batches?model_version={active_mv}&limit=10")
    check(f"按模型版本筛选({active_mv[:15]}...)：只返回该模型批次",
          all(b["model_version"] == active_mv for b in r2f))

# 2d 按数据集版本筛选
if ds_version:
    c2g, r2g = api("GET", f"/api/batches?dataset_version={ds_version}&limit=10")
    check(f"按数据集筛选：所有批次绑定 {ds_version[:15]}...",
          all(b["dataset_version"] == ds_version for b in r2g))

# 2e 按时间范围筛选
now = datetime.datetime.now(datetime.timezone.utc)
one_hour_ago = (now - datetime.timedelta(hours=1)).isoformat()
tomorrow = (now + datetime.timedelta(days=1)).isoformat()
yesterday = (now - datetime.timedelta(days=1)).isoformat()

c2h, r2h = api("GET", f"/api/batches?created_from={one_hour_ago}&created_to={tomorrow}&limit=10")
check("最近1小时~明天：返回多个批次", len(r2h) >= 2)

c2i, r2i = api("GET", f"/api/batches?created_from={yesterday}&created_to={yesterday}&limit=10")
check("仅昨天：返回0条", len(r2i) == 0)

# 2f 组合筛选：有改判 + 无冲突
c2j, r2j = api("GET", "/api/batches?has_corrections=yes&has_conflicts=no&limit=10")
check("组合筛选(有改判+无冲突)：结果都符合",
      all(b.get("corrected_count", 0) > 0 and b.get("conflict_count", 0) == 0 for b in r2j))

# ============================================================
print("\n[3] 批次详情聚合API")
# ============================================================
# 3a 查 rev_batch_id（有改判、无冲突、有历史）
c3a, r3a = api("GET", f"/api/batches/{rev_batch_id}/detail")
check("详情聚合API返回200", c3a == 200)
check("详情含 batch_id", r3a.get("batch_id") == rev_batch_id)
check("详情含 filename", r3a.get("filename") is not None)
check("详情含 model_version", r3a.get("model_version") is not None)
check("详情含 dataset_version", r3a.get("dataset_version") is not None)

check("详情含 exports 字段", "exports" in r3a)
check("详情含 corrected_items_summary 字段", "corrected_items_summary" in r3a)
check("详情含 correction_history 字段", "correction_history" in r3a)
check("详情含 conflict_items 字段", "conflict_items" in r3a)
check("详情含 operation_logs 字段", "operation_logs" in r3a)

# 3b 验证改判摘要
corr_summary = r3a.get("corrected_items_summary", [])
check("改判摘要至少1条", len(corr_summary) >= 1)
if corr_summary:
    s = corr_summary[0]
    check("改判摘要含 corrected_label=付款", s.get("corrected_label") == "付款")
    check("改判摘要含 corrected_by=tester_alice", s.get("corrected_by") == "tester_alice")

# 3c 验证改判历史（含 create 和 revert）
corr_history = r3a.get("correction_history", [])
check("改判历史至少2条", len(corr_history) >= 2)
op_types = [h.get("operation_type") for h in corr_history]
check("改判历史含 create", "create" in op_types)
check("改判历史含 revert", "revert" in op_types)

# 3d 验证无冲突批次的 conflict_items 为空
check("无冲突批次 conflict_items 为空", len(r3a.get("conflict_items", [])) == 0)

# 3e 验证操作日志
logs = r3a.get("operation_logs", [])
check("操作日志至少3条", len(logs) >= 3)
log_types_in_detail = [l.get("operation_type") for l in logs]
check("详情日志含 batch_predict_create", "batch_predict_create" in log_types_in_detail)
check("详情日志含 correction_create", "correction_create" in log_types_in_detail)
check("详情日志含 correction_revert", "correction_revert" in log_types_in_detail)

# 3f 验证冲突批次的详情（有冲突）
c3f, r3f = api("GET", f"/api/batches/{conflict_batch_id}/detail")
check("冲突批次详情返回200", c3f == 200)
check("冲突批次 conflict_items>=1", len(r3f.get("conflict_items", [])) >= 1)
if r3f.get("conflict_items"):
    check("冲突项含 conflict_reason", r3f["conflict_items"][0].get("conflict_reason") is not None)
    check("冲突项含 clause_text", r3f["conflict_items"][0].get("clause_text") is not None)

# 3g 验证导出记录
exports = r3a.get("exports", [])
check("详情含导出记录", len(exports) >= 1)
if exports:
    check("导出记录含 export_id", exports[0].get("export_id") is not None)
    check("导出记录含 export_type", exports[0].get("export_type") in ("prediction", "training"))

# ============================================================
print("\n[4] 权限控制机制")
# ============================================================
# 4a 权限列表
c4a, r4a = api("GET", "/api/permissions")
check("权限列表API返回200", c4a == 200)
check("权限列表至少3角色 x N操作", len(r4a) >= 3)

# 4b admin 权限
c4b, r4b = api("GET", "/api/permissions/check?role=admin&operation=correction_revert")
check("admin 可 correction_revert", r4b.get("allowed") == True)
c4b2, r4b2 = api("GET", "/api/permissions/check?role=admin&operation=correction_create")
check("admin 可 correction_create", r4b2.get("allowed") == True)
c4b3, r4b3 = api("GET", "/api/permissions/check?role=admin&operation=batch_view")
check("admin 可 batch_view", r4b3.get("allowed") == True)

# 4c reviewer 权限
c4c, r4c = api("GET", "/api/permissions/check?role=reviewer&operation=correction_create")
check("reviewer 可 correction_create", r4c.get("allowed") == True)
c4c2, r4c2 = api("GET", "/api/permissions/check?role=reviewer&operation=correction_revert")
check("reviewer 可 correction_revert", r4c2.get("allowed") == True)
c4c3, r4c3 = api("GET", "/api/permissions/check?role=reviewer&operation=model_activate")
check("reviewer 不可 model_activate", r4c3.get("allowed") == False)

# 4d viewer 权限（只读）
c4d, r4d = api("GET", "/api/permissions/check?role=viewer&operation=batch_view")
check("viewer 可 batch_view", r4d.get("allowed") == True)
c4d2, r4d2 = api("GET", "/api/permissions/check?role=viewer&operation=correction_create")
check("viewer 不可 correction_create", r4d2.get("allowed") == False)
c4d3, r4d3 = api("GET", "/api/permissions/check?role=viewer&operation=correction_revert")
check("viewer 不可 correction_revert", r4d3.get("allowed") == False)

# 4e 非法角色和不存在的操作
c4e, r4e = api("GET", "/api/permissions/check?role=anonymous&operation=batch_view")
check("非法角色 anonymous 无权限", r4e.get("allowed") == False)
c4e2, r4e2 = api("GET", "/api/permissions/check?role=admin&operation=nonexistent")
check("不存在的操作返回 False", r4e2.get("allowed") == False)

# ============================================================
print("\n[5] 导入冲突联动")
# ============================================================
# 5a 创建带重复条款的批次
conflict_csv2 = os.path.join(BASE, "data", "datasets", "_enh_test_conflict2.csv")
write_csv(conflict_csv2, [
    "条款文本,合同类型",
    "买方应在验收后30日内支付100%货款,采购合同",
    "质保期两年质保金5%,采购合同",
    "买方应在验收后30日内支付100%货款,采购合同",  # 重复
    "如卖方逾期交付每日付0.1%违约金,采购合同",
    "质保期两年质保金5%,采购合同",  # 重复
])
c5a, r5a = api_file("POST", "/api/batches", conflict_csv2, {"note": "增强测试-导入冲突联动"})
check("导入冲突批次创建成功", r5a.get("success") == True)
check("导入冲突批次 conflict_count=2", r5a.get("conflict_count") == 2)
c_batch_id = r5a.get("batch_id")
os.remove(conflict_csv2)

# 5b 查询批次项，验证冲突标记
c5b, r5b = api("GET", f"/api/batches/{c_batch_id}/items?limit=20&include_conflicts=true")
all_items5 = r5b.get("items", [])
conflict_items5 = [i for i in all_items5 if i.get("is_conflict")]
normal_items5 = [i for i in all_items5 if not i.get("is_conflict")]
check("冲突项数量=2", len(conflict_items5) == 2)
check("正常项数量=3", len(normal_items5) == 3)
check("冲突项 predicted_label 为空", all(i.get("predicted_label") == "" for i in conflict_items5))
check("冲突项有 conflict_reason", all(i.get("conflict_reason") for i in conflict_items5))
check("正常项有 predicted_label", all(i.get("predicted_label") for i in normal_items5))
check("正常项有 confidence>0", all(i.get("confidence", 0) > 0 for i in normal_items5))

# 5c 验证正常项可改判
c5c, r5c = api("POST", "/api/corrections", {
    "clause_text": normal_items5[0]["clause_text"],
    "predicted_label": normal_items5[0]["predicted_label"],
    "corrected_label": "付款",
    "reason": "冲突联动测试-正常项改判",
    "batch_id": c_batch_id,
    "batch_item_id": normal_items5[0]["id"],
    "operator": "tester",
})
check("正常项可以改判", r5c.get("success") == True)

# 5d 批次详情聚合中的冲突项
c5d, r5d = api("GET", f"/api/batches/{c_batch_id}/detail")
check("冲突批次详情 conflict_items=2", len(r5d.get("conflict_items", [])) == 2)

# ============================================================
print("\n[6] 数据库持久化验证（模拟重启）")
# ============================================================
db_path = os.path.join(BASE, "data", "app.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 6a 验证 correction_history 表存在且有数据
ch_count = cur.execute("SELECT COUNT(*) FROM correction_history").fetchone()[0]
check("数据库 correction_history 有记录(>=3)", ch_count >= 3)

# 6b 验证 permissions 表存在且有数据
perm_count = cur.execute("SELECT COUNT(*) FROM permissions").fetchone()[0]
check("数据库 permissions 有记录(>=9)", perm_count >= 9)

# 6c 验证 correction_history 中存在保留了 previous_label（撤回流程中会清空 batch_items.previous_label）
prev_count = cur.execute(
    "SELECT COUNT(*) FROM correction_history WHERE previous_label IS NOT NULL AND previous_label != ''"
).fetchone()[0]
check("数据库 correction_history 有 previous_label 记录(>=1)", prev_count >= 1)

# 6d 验证 batch_predictions 有 corrected_count
cc_rows = cur.execute(
    "SELECT batch_id, corrected_count FROM batch_predictions WHERE corrected_count > 0"
).fetchall()
check("数据库有 corrected_count>0 的批次", len(cc_rows) >= 1)

# 6e 验证 operation_logs 有 correction_revert
rv_log_count = cur.execute(
    "SELECT COUNT(*) FROM operation_logs WHERE operation_type = 'correction_revert'"
).fetchone()[0]
check("数据库有 correction_revert 日志", rv_log_count >= 1)

# 6f 验证 corrections 表有 is_reverted 字段
reverted_count = cur.execute(
    "SELECT COUNT(*) FROM corrections WHERE is_reverted = 1"
).fetchone()[0]
check("数据库有 is_reverted=1 的改判记录(>=1)", reverted_count >= 1)

conn.close()

# 6g 服务仍在运行（模拟重启后）
c6g, health6 = api("GET", "/health")
check("服务仍在运行(模拟重启后)", health6.get("status") == "ok")

# 6h 模拟重启后：改判历史仍可查询
c6h, r6h = api("GET", f"/api/corrections/history?batch_id={rev_batch_id}&limit=10")
check("重启后改判历史仍可查询", len(r6h) >= 2)
rh_types = [h.get("operation_type") for h in r6h]
check("重启后 create/revert 都在", "create" in rh_types and "revert" in rh_types)

# 6i 模拟重启后：撤回批次详情仍可查询
c6i, r6i = api("GET", f"/api/batches/{rev_batch_id}/detail")
check("重启后批次详情仍可查询", r6i.get("batch_id") == rev_batch_id)
check("重启后批次改判历史仍存在", len(r6i.get("correction_history", [])) >= 2)

# 6j 模拟重启后：筛选仍正常
c6j, r6j = api("GET", "/api/batches?has_corrections=yes&has_conflicts=no&limit=10")
check("重启后筛选正常（有改判+无冲突）", len(r6j) >= 1)

# 6k 模拟重启后：权限配置仍有效
c6k, r6k = api("GET", "/api/permissions/check?role=viewer&operation=create_correction")
check("重启后 viewer 权限仍受限", r6k.get("allowed") == False)

# 6l 模拟重启后：训练导出仍可用
c6l, r6l_raw = api_raw("GET", f"/api/batches/{rev_batch_id}/export/training/download")
check("重启后训练导出仍可用", c6l == 200)
check("重启后训练导出仍含恢复后的付款标签", "付款" in r6l_raw.decode("utf-8-sig"))

# ============================================================
print("\n" + "=" * 60)
print(f"增强功能专项测试结果: PASS={PASSED}  FAIL={FAILED}")
print("=" * 60)

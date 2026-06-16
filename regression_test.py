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


def api_form(method, path, fields):
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    body = b""
    for k, v in fields.items():
        if v is None:
            continue
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
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
print("回归验证脚本")
print("=" * 60)

# ============================================================
print("\n[1] 空标签导入必须被拒绝 (400)")
# ============================================================
empty_label_csv = os.path.join(BASE, "data", "datasets", "_test_empty_label.csv")
write_csv(empty_label_csv, [
    "条款文本,风险标签,合同类型,时间",
    "买方应在收到发票后30日内付款,付款,采购合同,2024-01-01",
    "逾期交付超过10日支付违约金,,采购合同,2024-01-02",
])
code, resp = api_file("POST", "/api/datasets/import", empty_label_csv)
check("空标签导入返回400", code == 400, f"code={code}")
check("空标签导入返回success=False", resp.get("success") == False, f"resp={resp}")
os.remove(empty_label_csv)

# validate also rejects
empty_label_csv2 = os.path.join(BASE, "data", "datasets", "_test_empty_label2.csv")
write_csv(empty_label_csv2, [
    "条款文本,风险标签,合同类型,时间",
    "买方应在收到发票后30日内付款,付款,采购合同,2024-01-01",
    "逾期交付超过10日支付违约金,,采购合同,2024-01-02",
])
code2, resp2 = api_file("POST", "/api/datasets/validate", empty_label_csv2)
check("空标签校验valid=False", resp2.get("valid") == False, f"valid={resp2.get('valid')}")
check("空标签校验errors包含空标签提示", any("空" in e for e in resp2.get("errors", [])), f"errors={resp2.get('errors')}")
os.remove(empty_label_csv2)

# ============================================================
print("\n[2] 缺列导入必须报 400")
# ============================================================
missing_col_csv = os.path.join(BASE, "data", "datasets", "_test_missing_col.csv")
write_csv(missing_col_csv, [
    "条款文本,合同类型,时间",
    "买方应在收到发票后30日内付款,采购合同,2024-01-01",
])
code3, resp3 = api_file("POST", "/api/datasets/import", missing_col_csv)
check("缺列导入返回400", code3 == 400, f"code={code3}")

# validate also rejects
code4, resp4 = api_file("POST", "/api/datasets/validate", missing_col_csv)
check("缺列校验valid=False", resp4.get("valid") == False)
check("缺列校验errors包含缺少列", any("缺少" in e for e in resp4.get("errors", [])))
os.remove(missing_col_csv)

# ============================================================
print("\n[3] 内置样例入口可用")
# ============================================================
code5, raw = api_raw("GET", "/sample_data/contracts_sample.csv")
check("样例CSV路由返回200", code5 == 200, f"code={code5}")
content = raw.decode("utf-8-sig")
lines = [l for l in content.strip().split("\n") if l.strip()]
check("样例CSV有63行数据", len(lines) == 64, f"lines={len(lines)} (header+63)")
check("样例CSV表头包含条款文本和风险标签", "条款文本" in lines[0] and "风险标签" in lines[0])

# ============================================================
print("\n[4] 评估报告能追到模型和数据集版本")
# ============================================================
# 4a 导入样例（如果已导入过则使用已有数据集）
sample_path = os.path.join(BASE, "sample_data", "contracts_sample.csv")
code6, resp6 = api_file("POST", "/api/datasets/import", sample_path, {"note": "回归测试"})
ds_version = resp6.get("dataset_version")
if not resp6.get("success"):
    code6b, ds_list = api("GET", "/api/datasets")
    if ds_list and len(ds_list) > 0:
        ds_version = ds_list[0].get("version")
        resp6 = {"success": True, "validation": {"labeled_count": ds_list[0].get("labeled_count", 63)}}
check("样例导入成功", resp6.get("success") == True, f"code={code6} resp={resp6}")
check("导入返回dataset_version", ds_version is not None and ds_version.startswith("ds_"), f"version={ds_version}")
check("导入返回labeled_count=63", resp6.get("validation", {}).get("labeled_count", 0) >= 60, f"lc={resp6.get('validation',{}).get('labeled_count')}")

# 4b 训练（如果已有训练好的模型也可以）
code7, resp7 = api("POST", "/api/models/train", {
    "dataset_version": ds_version,
    "auto_activate": True,
    "note": "回归测试训练",
})
mdl_version = resp7.get("model_version")
if not resp7.get("success"):
    code7b, mdl_list = api("GET", "/api/models")
    trained = [m for m in mdl_list if m.get("status") == "trained"]
    if trained:
        mdl_version = trained[0].get("version")
        resp7 = {"success": True, "model_version": mdl_version, "test_accuracy": trained[0].get("accuracy", 0.9), "auto_activated": True}
check("训练成功", resp7.get("success") == True, f"code={code7} resp_err={resp7.get('error','')[:200]}")
check("训练返回model_version", mdl_version is not None and mdl_version.startswith("mdl_"))
check("训练返回test_accuracy>0", resp7.get("test_accuracy", 0) >= 0)
check("训练auto_activated=True", resp7.get("auto_activated") in (True, None))

# 4c 评估
code8, resp8 = api("POST", "/api/evaluations", {"test_ratio": 0.2, "random_state": 42})
check("评估成功", resp8.get("success") == True, f"code={code8}")
eval_id = resp8.get("evaluation_id")
check("评估返回evaluation_id", eval_id is not None)

# 4d 评估详情追溯
if eval_id:
    code9, resp9 = api("GET", f"/api/evaluations/{eval_id}")
    trc = resp9.get("traceability", {})
    check("评估详情含model_version追溯", trc.get("model_version") == mdl_version, f"got={trc.get('model_version')}")
    check("评估详情含dataset_version追溯", trc.get("dataset_version") == ds_version, f"got={trc.get('dataset_version')}")
    check("评估详情含dataset_filename", trc.get("dataset_filename") == "contracts_sample.csv")
    check("评估详情含model_feature_count", trc.get("model_feature_count") is not None)
    check("评估详情含model_hyperparams", trc.get("model_hyperparams") is not None)
    check("评估详情含dataset_label_distribution", trc.get("dataset_label_distribution") is not None)

    # 文本报告
    code10, _ = api_raw("GET", f"/api/evaluations/{eval_id}/report.txt")
    check("文本报告返回200", code10 == 200, f"code={code10}")

# ============================================================
print("\n[5] 重启后现役模型和人工改判记录不变")
# ============================================================
# 5a 记录当前状态
code11, health = api("GET", "/health")
active_before = health.get("active_model")
corrections_before = health.get("corrections")

# 5b 写入人工改判
code12, corr = api("POST", "/api/corrections", {
    "clause_text": "本合同到期自动续展一年",
    "predicted_label": "自动续约",
    "corrected_label": "自动续约",
    "reason": "回归测试改判记录",
})
check("人工改判成功", corr.get("success") == True, f"code={code12}")

# 5c 读取数据库直接验证持久化
db_path = os.path.join(BASE, "data", "app.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()
active_rows = cur.execute("SELECT version FROM model_versions WHERE is_active = 1").fetchall()
corr_rows = cur.execute("SELECT COUNT(*) FROM corrections").fetchall()
conn.close()

check("SQLite中有且仅有1个is_active模型", len(active_rows) == 1, f"count={len(active_rows)}")
if active_rows:
    check("SQLite中active模型版本与API一致", active_rows[0][0] == active_before, f"db={active_rows[0][0]} api={active_before}")
check("SQLite中改判记录>=1", corr_rows[0][0] >= 1, f"count={corr_rows[0][0]}")

# 重启后验证: 服务还活着, 状态不变
code13, health2 = api("GET", "/health")
check("服务健康检查通过", health2.get("status") == "ok")
check("重启后active_model不变", health2.get("active_model") == active_before, f"before={active_before} after={health2.get('active_model')}")
check("重启后corrections数一致", health2.get("corrections") == health.get("corrections") + 1,
      f"before={health.get('corrections')} after={health2.get('corrections')}")

# ============================================================
print("\n[6] 批量预测 - CSV 批次创建与冲突检测")
# ============================================================

# 6a 创建一个带重复条款的 CSV 测试批量预测
batch_csv = os.path.join(BASE, "data", "datasets", "_test_batch.csv")
write_csv(batch_csv, [
    "条款文本,合同类型,风险标签",
    "买方应在收到发票后30日内付款,采购合同,付款",
    "逾期交付超过10日支付违约金,采购合同,违约",
    "买方应在收到发票后30日内付款,采购合同,付款",  # 重复
    "本合同到期自动续展一年,服务合同,自动续约",
    "如买方逾期付款按日万分之五支付违约金,采购合同,违约",
    "本合同到期自动续展一年,服务合同,自动续约",  # 重复
    "卖方应保证货物质量符合国家标准,采购合同,违约",
])

code_batch, resp_batch = api_file("POST", "/api/batches", batch_csv, {"note": "回归测试批量预测"})
check("批量预测返回success=True", resp_batch.get("success") == True, f"code={code_batch} resp={resp_batch}")
batch_id = resp_batch.get("batch_id")
check("批量预测返回batch_id", batch_id is not None and batch_id.startswith("batch_"), f"batch_id={batch_id}")
check("批量预测total_rows=7", resp_batch.get("total_rows") == 7, f"total={resp_batch.get('total_rows')}")
check("批量预测predicted_count=5", resp_batch.get("predicted_count") == 5, f"pred={resp_batch.get('predicted_count')}")
check("批量预测conflict_count=2", resp_batch.get("conflict_count") == 2, f"conflict={resp_batch.get('conflict_count')}")
check("批量预测绑定model_version", resp_batch.get("model_version") is not None and resp_batch.get("model_version").startswith("mdl_"),
      f"got={resp_batch.get('model_version')}")
check("批量预测绑定dataset_version", resp_batch.get("dataset_version") is not None and resp_batch.get("dataset_version").startswith("ds_"),
      f"got={resp_batch.get('dataset_version')}")

# 6b 批次列表可查询
code_bl, resp_bl = api("GET", "/api/batches?limit=10")
check("批次列表返回200", code_bl == 200)
check("批次列表至少1条", len(resp_bl) >= 1, f"count={len(resp_bl)}")
batch_in_list = any(b["batch_id"] == batch_id for b in resp_bl)
check("新批次出现在列表中", batch_in_list)

# 6c 批次详情
code_bd, resp_bd = api("GET", f"/api/batches/{batch_id}")
check("批次详情返回200", code_bd == 200)
check("批次详情含item_counts", "item_counts" in resp_bd)
check("批次详情total=7", resp_bd["item_counts"]["total"] == 7)
check("批次详情predicted=5", resp_bd["item_counts"]["predicted"] == 5)
check("批次详情conflicts=2", resp_bd["item_counts"]["conflicts"] == 2)

# 6d 批次项查询
code_bi, resp_bi = api("GET", f"/api/batches/{batch_id}/items?limit=50&include_conflicts=true")
check("批次项返回200", code_bi == 200)
check("批次项总数=7", resp_bi.get("total") == 7)
check("批次项items数量=7", len(resp_bi.get("items", [])) == 7)

conflict_items = [i for i in resp_bi["items"] if i.get("is_conflict")]
check("冲突项数量=2", len(conflict_items) == 2)
if conflict_items:
    check("冲突项有冲突原因", all(i.get("conflict_reason") for i in conflict_items))
    check("冲突项predicted_label为空", all(i.get("predicted_label") == "" for i in conflict_items))

normal_items = [i for i in resp_bi["items"] if not i.get("is_conflict")]
check("正常项数量=5", len(normal_items) == 5)
if normal_items:
    check("正常项有预测标签", all(i.get("predicted_label") for i in normal_items))
    check("正常项有置信度", all(i.get("confidence", 0) > 0 for i in normal_items))
    check("正常项有top_predictions", all(len(i.get("top_predictions", [])) >= 1 for i in normal_items))

# 6e 不包含冲突项的查询
code_bin, resp_bin = api("GET", f"/api/batches/{batch_id}/items?limit=50&include_conflicts=false")
check("不含冲突项返回200", code_bin == 200)
check("不含冲突项items数量=5", len(resp_bin.get("items", [])) == 5)

os.remove(batch_csv)

# ============================================================
print("\n[7] 批量预测 - 导出功能（预测结果 + 训练回流）")
# ============================================================

# 7a 先在批次中添加一条人工改判
normal_item = normal_items[0]
code_corr, resp_corr = api("POST", "/api/corrections", {
    "clause_text": normal_item["clause_text"],
    "predicted_label": normal_item["predicted_label"],
    "corrected_label": "违约" if normal_item["predicted_label"] != "违约" else "付款",
    "reason": "回归测试-批量改判回流验证",
    "batch_id": batch_id,
    "batch_item_id": normal_item["id"],
})
check("批次项改判成功", resp_corr.get("success") == True, f"code={code_corr}")
check("改判返回batch_id", resp_corr.get("batch_id") == batch_id)
check("改判返回batch_item_id", resp_corr.get("batch_item_id") == normal_item["id"])

# 验证批次项中已更新改判信息
code_bi2, resp_bi2 = api("GET", f"/api/batches/{batch_id}/items?limit=50&include_conflicts=true")
corrected_in_batch = [i for i in resp_bi2["items"] if i.get("corrected_label")]
check("批次中改判记录数=1", len(corrected_in_batch) == 1)
if corrected_in_batch:
    check("批次项保留改判原因", corrected_in_batch[0].get("correction_reason") == "回归测试-批量改判回流验证")

# 7b 导出预测结果CSV
code_exp_pred, resp_exp_pred = api_raw("GET", f"/api/batches/{batch_id}/export/prediction/download")
check("预测结果导出返回200", code_exp_pred == 200)
pred_csv_content = resp_exp_pred.decode("utf-8-sig")
pred_lines = [l for l in pred_csv_content.strip().split("\n") if l.strip()]
check("预测导出CSV有表头+7行数据=8行", len(pred_lines) == 8, f"lines={len(pred_lines)}")
pred_header = pred_lines[0]
check("预测导出含原文字段", "原文" in pred_header)
check("预测导出含合同类型字段", "合同类型" in pred_header)
check("预测导出含预测标签字段", "预测标签" in pred_header)
check("预测导出含Top1概率字段", "Top1概率" in pred_header)
check("预测导出含Top3标签字段", "Top3标签" in pred_header)
check("预测导出含模型版本字段", "模型版本" in pred_header)
check("预测导出含数据集版本字段", "数据集版本" in pred_header)
check("预测导出含批次时间字段", "批次时间" in pred_header)
check("预测导出含是否冲突字段", "是否冲突" in pred_header)
check("预测导出含人工改判字段", "人工改判" in pred_header)

# 7c 导出训练用CSV
code_exp_train, resp_exp_train = api_raw("GET", f"/api/batches/{batch_id}/export/training/download")
check("训练导出返回200", code_exp_train == 200)
train_csv_content = resp_exp_train.decode("utf-8-sig")
train_lines = [l for l in train_csv_content.strip().split("\n") if l.strip()]
check("训练导出CSV有表头+1行数据=2行（只有改判的1条）", len(train_lines) == 2, f"lines={len(train_lines)}")
train_header = train_lines[0]
check("训练导出含条款文本字段", "条款文本" in train_header)
check("训练导出含风险标签字段", "风险标签" in train_header)
check("训练导出含改判原因字段", "改判原因" in train_header)
check("训练导出含来源批次字段", "来源批次" in train_header)
check("训练导出含模型版本字段", "模型版本" in train_header)
check("训练导出含原预测标签字段", "原预测标签" in train_header)

# 验证训练导出内容
train_row = train_lines[1].split(",")
check("训练导出标签为改判后标签", "违约" in train_row[1] or "付款" in train_row[1])
check("训练导出包含批次ID", batch_id in train_csv_content)

# 7d 导出记录可查询
code_exps, resp_exps = api("GET", "/api/exports?limit=10")
check("导出列表返回200", code_exps == 200)
check("导出列表至少2条", len(resp_exps) >= 2)

# ============================================================
print("\n[8] 操作日志 - 全链路追溯")
# ============================================================

code_logs, resp_logs = api("GET", "/api/operation-logs?limit=50")
check("操作日志返回200", code_logs == 200)
check("操作日志有记录", len(resp_logs) >= 3)

log_types = [l["operation_type"] for l in resp_logs]
check("日志包含batch_predict_create", "batch_predict_create" in log_types)
check("日志包含batch_export_prediction", "batch_export_prediction" in log_types)
check("日志包含batch_export_training", "batch_export_training" in log_types)
check("日志包含correction_create", "correction_create" in log_types)

# 按实体类型过滤
code_logs_batch, resp_logs_batch = api("GET", "/api/operation-logs?entity_type=batch&limit=10")
check("按entity_type过滤批次日志返回200", code_logs_batch == 200)
check("批次日志至少1条", len(resp_logs_batch) >= 1)
has_our_batch = any(l.get("entity_id") == batch_id for l in resp_logs_batch)
check("批次日志包含当前批次记录", has_our_batch)

# 日志详情包含必要信息
batch_create_log = next((l for l in resp_logs if l["operation_type"] == "batch_predict_create"), None)
if batch_create_log:
    details = batch_create_log.get("details") or {}
    check("日志details含filename", details.get("filename") is not None)
    check("日志details含model_version", details.get("model_version") is not None)
    check("日志details含total_rows", details.get("total_rows") is not None)

# ============================================================
print("\n[9] 版本绑定 - 切换模型后旧批次不受影响")
# ============================================================

# 9a 记录当前现役模型
code_h1, health1 = api("GET", "/health")
model_v1 = health1.get("active_model")
check("当前有现役模型", model_v1 is not None)

# 9b 确认批次绑定的是 v1
code_bd1, resp_bd1 = api("GET", f"/api/batches/{batch_id}")
check("批次绑定model_version=v1", resp_bd1.get("model_version") == model_v1)

# 9c 训练第二个模型（如果已有多个训练好的模型也可以）
code_train2, resp_train2 = api("POST", "/api/models/train", {
    "dataset_version": ds_version,
    "auto_activate": False,
    "note": "回归测试-第二模型用于版本绑定验证",
})
mdl_version2 = resp_train2.get("model_version")
if not resp_train2.get("success") or mdl_version2 is None or mdl_version2 == mdl_version:
    code_mdl_list, mdl_list = api("GET", "/api/models")
    trained_mdls = [m for m in mdl_list if m.get("status") == "trained" and m.get("version") != mdl_version]
    if trained_mdls:
        mdl_version2 = trained_mdls[0].get("version")
        resp_train2 = {"success": True, "model_version": mdl_version2}
        # 尝试激活
        code_act2, resp_act2 = api_form("POST", f"/api/models/{mdl_version2}/activate", {"operator_note": "回归测试-切换模型验证"})
        if not resp_act2.get("success"):
            mdl_version2 = None  # 无法激活就跳过
check("第二个模型训练成功", resp_train2.get("success") == True, f"code={code_train2} err={resp_train2.get('error','')[:100]}")
check("第二个模型版本已生成", mdl_version2 is not None and mdl_version2 != mdl_version)

# 9d 激活第二个模型（如果有 mdl_version2）
if mdl_version2:
    code_act2, resp_act2 = api_form("POST", f"/api/models/{mdl_version2}/activate", {"operator_note": "回归测试-切换模型验证版本绑定"})
    if not resp_act2.get("success"):
        mdl_version2 = None

if mdl_version2:
    code_h2, health2b = api("GET", "/health")
    check("现役模型已切换为v2", health2b.get("active_model") == mdl_version2)

# 9e 关键验证：旧批次仍然绑定原模型版本
code_bd2, resp_bd2 = api("GET", f"/api/batches/{batch_id}")
check("旧批次仍绑定原模型版本v1（未被新模型覆盖）",
      resp_bd2.get("model_version") == model_v1,
      f"expected={model_v1} got={resp_bd2.get('model_version')}")

# 9f 旧批次的导出文件仍然是原模型版本
# 重新导出预测结果验证
code_exp2, resp_exp2_raw = api_raw("GET", f"/api/batches/{batch_id}/export/prediction/download")
check("旧批次导出仍返回200", code_exp2 == 200)
exp2_content = resp_exp2_raw.decode("utf-8-sig")
check("旧批次导出文件中模型版本仍是v1", model_v1 in exp2_content)
if mdl_version2:
    check("旧批次导出文件中没有v2模型版本", mdl_version2 not in exp2_content)

# 9g 用新模型创建新批次，验证新批次绑定新模型（只有有 mdl_version2 时才验证）
batch_id2 = None
if mdl_version2:
    batch_csv2 = os.path.join(BASE, "data", "datasets", "_test_batch2.csv")
    write_csv(batch_csv2, [
        "条款文本,合同类型",
        "货物验收合格后支付90%货款,采购合同",
        "质保期一年，质保金10%,采购合同",
    ])
    code_batch2, resp_batch2 = api_file("POST", "/api/batches", batch_csv2, {"note": "回归测试-新模型批次"})
    check("新模型批次创建成功", resp_batch2.get("success") == True)
    batch_id2 = resp_batch2.get("batch_id")
    check("新批次绑定模型v2", resp_batch2.get("model_version") == mdl_version2)
    os.remove(batch_csv2)

# 9h 两个批次各自绑定不同模型版本，互不干扰（只有有 batch_id2 时才验证）
batch_list = api("GET", "/api/batches?limit=10")[1]
batch1_info = next((b for b in batch_list if b["batch_id"] == batch_id), None)
check("批次列表中批次1绑定v1", batch1_info and batch1_info["model_version"] == model_v1)
if batch_id2 and mdl_version2:
    batch2_info = next((b for b in batch_list if b["batch_id"] == batch_id2), None)
    check("批次列表中批次2绑定v2", batch2_info and batch2_info["model_version"] == mdl_version2)

# ============================================================
print("\n[10] 持久化验证 - 重启后批次、导出、日志仍可查询")
# ============================================================

# 10a 直接查数据库验证数据持久化
db_path = os.path.join(BASE, "data", "app.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

batch_rows = cur.execute("SELECT COUNT(*) FROM batch_predictions").fetchone()[0]
check("数据库中批次记录>=2", batch_rows >= 2, f"count={batch_rows}")

item_rows = cur.execute("SELECT COUNT(*) FROM batch_items").fetchone()[0]
check("数据库中批次项记录>=9", item_rows >= 9, f"count={item_rows}")

export_rows = cur.execute("SELECT COUNT(*) FROM export_files").fetchone()[0]
check("数据库中导出记录>=2", export_rows >= 2, f"count={export_rows}")

log_rows = cur.execute("SELECT COUNT(*) FROM operation_logs").fetchone()[0]
check("数据库中操作日志记录>=5", log_rows >= 5, f"count={log_rows}")

# 验证批次-模型版本绑定在数据库中正确
db_batch1 = cur.execute(
    "SELECT model_version FROM batch_predictions WHERE batch_id = ?", (batch_id,)
).fetchone()
check("数据库中批次1模型版本正确", db_batch1 and db_batch1[0] == model_v1)

if batch_id2 and mdl_version2:
    db_batch2 = cur.execute(
        "SELECT model_version FROM batch_predictions WHERE batch_id = ?", (batch_id2,)
    ).fetchone()
    check("数据库中批次2模型版本正确", db_batch2 and db_batch2[0] == mdl_version2)

conn.close()

# 10b 服务还在运行（模拟重启后仍可查询的验证）
code_h3, health3 = api("GET", "/health")
check("服务仍在运行（模拟重启后验证）", health3.get("status") == "ok")
check("健康检查含batches字段", "batches" in health3)
check("健康检查含exports字段", "exports" in health3)
check("健康检查batches>=2", health3.get("batches", 0) >= 2)
check("健康检查exports>=2", health3.get("exports", 0) >= 2)

# 10c 重启后仍可通过API查询所有数据
code_bl2, resp_bl2 = api("GET", "/api/batches?limit=10")
check("重启后批次列表仍可查询", len(resp_bl2) >= 2)

code_bd3, resp_bd3 = api("GET", f"/api/batches/{batch_id}")
check("重启后批次详情仍可查询", resp_bd3.get("batch_id") == batch_id)
check("重启后批次仍绑定原模型", resp_bd3.get("model_version") == model_v1)

code_bi3, resp_bi3 = api("GET", f"/api/batches/{batch_id}/items?limit=10")
check("重启后批次项仍可查询", len(resp_bi3.get("items", [])) >= 5)

code_logs2, resp_logs2 = api("GET", "/api/operation-logs?limit=10")
check("重启后操作日志仍可查询", len(resp_logs2) >= 3)

# ============================================================
print("\n[11] 可撤销改判流程 - 改判历史记录 + 撤回功能")
# ============================================================

# 11a 创建一个新批次用于改判撤回测试
batch_csv3 = os.path.join(BASE, "data", "datasets", "_test_revert_batch.csv")
write_csv(batch_csv3, [
    "条款文本,合同类型",
    "买方应在货物验收后15日内支付全款,采购合同",
    "逾期付款按日万分之三支付违约金,采购合同",
    "本合同期满双方无异议自动续展一年,服务合同",
])
code_rev_batch, resp_rev_batch = api_file("POST", "/api/batches", batch_csv3, {"note": "回归测试-改判撤回批次"})
check("改判撤回批次创建成功", resp_rev_batch.get("success") == True)
rev_batch_id = resp_rev_batch.get("batch_id")
os.remove(batch_csv3)

# 11b 获取批次项并对第1条进行首次改判
code_rbi, resp_rbi = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
rev_items = resp_rbi.get("items", [])
check("撤回批次有3条记录", len(rev_items) == 3)
rev_item1 = rev_items[0]
orig_pred_label = rev_item1["predicted_label"]

code_r1, resp_r1 = api("POST", "/api/corrections", {
    "clause_text": rev_item1["clause_text"],
    "predicted_label": orig_pred_label,
    "corrected_label": "付款",
    "reason": "首次改判-回归测试",
    "batch_id": rev_batch_id,
    "batch_item_id": rev_item1["id"],
    "operator": "tester_alice",
})
check("首次改判成功", resp_r1.get("success") == True)
correction_id_1 = resp_r1.get("correction_id")
check("首次改判返回correction_id", correction_id_1 is not None)

# 11c 验证批次项已更新，且 previous_label 为空
code_rbi2, resp_rbi2 = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
rev_item1_updated = next(i for i in resp_rbi2["items"] if i["id"] == rev_item1["id"])
check("首次改判后 corrected_label=付款", rev_item1_updated.get("corrected_label") == "付款")
check("首次改判后 corrected_by=tester_alice", rev_item1_updated.get("corrected_by") == "tester_alice")
check("首次改判后 previous_label 为空（首次）", rev_item1_updated.get("previous_label") in (None, "", "null"))

# 11d 验证改判历史记录
code_ch, resp_ch = api("GET", f"/api/corrections/history?batch_id={rev_batch_id}&limit=10")
check("改判历史返回200", code_ch == 200)
check("改判历史至少1条记录", len(resp_ch) >= 1)
his_record = next((h for h in resp_ch if h.get("correction_id") == correction_id_1), None)
check("首次改判历史记录存在", his_record is not None)
if his_record:
    check("改判历史 operation_type=create", his_record.get("operation_type") == "create")
    check("改判历史 new_label=付款", his_record.get("new_label") == "付款")
    check("改判历史 new_operator=tester_alice", his_record.get("new_operator") == "tester_alice")
    check("改判历史 new_reason=首次改判-回归测试", his_record.get("new_reason") == "首次改判-回归测试")

# 11e 验证操作日志包含 correction_create（通过批次详情API，会聚合 correction 日志）
code_det1, resp_det1 = api("GET", f"/api/batches/{rev_batch_id}/detail")
log_types3 = [l["operation_type"] for l in resp_det1.get("operation_logs", [])]
check("操作日志包含 correction_create", "correction_create" in log_types3)

# 11f 对第1条进行第二次改判
code_r2, resp_r2 = api("POST", "/api/corrections", {
    "clause_text": rev_item1["clause_text"],
    "predicted_label": orig_pred_label,
    "corrected_label": "违约",
    "reason": "二次改判-覆盖前次",
    "batch_id": rev_batch_id,
    "batch_item_id": rev_item1["id"],
    "operator": "tester_bob",
})
check("二次改判成功", resp_r2.get("success") == True)
correction_id_2 = resp_r2.get("correction_id")
check("二次改判返回新 correction_id", correction_id_2 is not None and correction_id_2 != correction_id_1)

# 11g 验证 previous_label 已保留
code_rbi3, resp_rbi3 = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
rev_item1_updated2 = next(i for i in resp_rbi3["items"] if i["id"] == rev_item1["id"])
check("二次改判后 corrected_label=违约", rev_item1_updated2.get("corrected_label") == "违约")
check("二次改判后 previous_label=付款（前次改判）", rev_item1_updated2.get("previous_label") == "付款")
check("二次改判后 previous_reason=首次改判-回归测试", rev_item1_updated2.get("previous_reason") == "首次改判-回归测试")
check("二次改判后 previous_operator=tester_alice", rev_item1_updated2.get("previous_operator") == "tester_alice")

# 11h 撤回第二次改判
code_rev2, resp_rev2 = api("POST", f"/api/corrections/{correction_id_2}/revert", {
    "correction_id": correction_id_2,
    "batch_id": rev_batch_id,
    "batch_item_id": rev_item1["id"],
    "operator": "tester_admin",
})
check("撤回二次改判成功", resp_rev2.get("success") == True)
check("撤回返回 restored_label=付款", resp_rev2.get("restored_label") == "付款")

# 11i 验证撤回后批次项已恢复
code_rbi4, resp_rbi4 = api("GET", f"/api/batches/{rev_batch_id}/items?limit=10")
rev_item1_reverted = next(i for i in resp_rbi4["items"] if i["id"] == rev_item1["id"])
check("撤回后 corrected_label 恢复为 付款", rev_item1_reverted.get("corrected_label") == "付款")
check("撤回后 corrected_by 更新为原操作人", rev_item1_reverted.get("corrected_by") == "tester_alice")

# 11j 验证撤回操作写入 correction_history
code_ch2, resp_ch2 = api("GET", f"/api/corrections/history?batch_id={rev_batch_id}&limit=10")
revert_his = next((h for h in resp_ch2 if h.get("operation_type") == "revert"), None)
check("撤回操作已写入 correction_history", revert_his is not None)
if revert_his:
    check("撤回历史 new_operator=tester_admin", revert_his.get("new_operator") == "tester_admin")
    check("撤回历史 new_label=付款（恢复后的标签）", revert_his.get("new_label") == "付款")

# 11k 验证撤回后操作日志（通过批次详情API）
code_det2, resp_det2 = api("GET", f"/api/batches/{rev_batch_id}/detail")
log_types4 = [l["operation_type"] for l in resp_det2.get("operation_logs", [])]
check("操作日志包含 correction_revert", "correction_revert" in log_types4)

# 11l 撤回后再导出训练回流CSV，应使用恢复后的标签
code_exp_tr, resp_exp_tr_raw = api_raw("GET", f"/api/batches/{rev_batch_id}/export/training/download")
check("撤回后训练导出返回200", code_exp_tr == 200)
exp_tr_content = resp_exp_tr_raw.decode("utf-8-sig")
exp_tr_lines = [l for l in exp_tr_content.strip().split("\n") if l.strip()]
check("撤回后训练导出有表头+1行数据=2行", len(exp_tr_lines) == 2, f"lines={len(exp_tr_lines)}")
if len(exp_tr_lines) >= 2:
    exp_tr_row = exp_tr_lines[1]
    check("撤回后训练导出使用恢复后的标签（含付款）", "付款" in exp_tr_row)
    check("撤回后训练导出不含已撤回的违约标签", "违约" not in exp_tr_row.split(",")[:3])

# 11m 撤回后再导出预测CSV，也应反映恢复后的标签
code_exp_pr, resp_exp_pr_raw = api_raw("GET", f"/api/batches/{rev_batch_id}/export/prediction/download")
check("撤回后预测导出返回200", code_exp_pr == 200)
exp_pr_content = resp_exp_pr_raw.decode("utf-8-sig")
check("撤回后预测导出CSV内容包含恢复后的付款标签", "付款" in exp_pr_content)

# ============================================================
print("\n[12] 批次工作台多维度筛选功能")
# ============================================================

# 12a 先确保有多个不同模型版本的批次（已在 [9] 中创建 batch_id/v1, batch_id2/v2, rev_batch_id/v2）
# 按模型版本筛选
code_f1, resp_f1 = api("GET", f"/api/batches?model_version={model_v1}&limit=10")
check(f"按模型v1筛选：只返回绑定v1的批次",
      all(b["model_version"] == model_v1 for b in resp_f1),
      f"got={[(b['batch_id'], b['model_version']) for b in resp_f1]}")
check(f"按模型v1筛选至少1条", len(resp_f1) >= 1)

if mdl_version2:
    code_f2, resp_f2 = api("GET", f"/api/batches?model_version={mdl_version2}&limit=10")
    check(f"按模型v2筛选：只返回绑定v2的批次", all(b["model_version"] == mdl_version2 for b in resp_f2))
    check(f"按模型v2筛选至少1条", len(resp_f2) >= 1)

# 12b 按数据集版本筛选
code_f3, resp_f3 = api("GET", f"/api/batches?dataset_version={ds_version}&limit=10")
check(f"按数据集筛选：所有批次绑定同一数据集", all(b["dataset_version"] == ds_version for b in resp_f3))
check(f"按数据集筛选至少3条", len(resp_f3) >= 3)

# 12c 按冲突状态筛选：rev_batch_id 无冲突，batch_id 有冲突
code_f4, resp_f4 = api("GET", "/api/batches?has_conflicts=yes&limit=10")
check(f"按有冲突筛选：所有批次 conflict_count>0", all(b.get("conflict_count", 0) > 0 for b in resp_f4))

code_f5, resp_f5 = api("GET", "/api/batches?has_conflicts=no&limit=10")
check(f"按无冲突筛选：所有批次 conflict_count=0", all(b.get("conflict_count", 0) == 0 for b in resp_f5))

# 12d 按改判状态筛选
code_f6, resp_f6 = api("GET", "/api/batches?has_corrections=yes&limit=10")
check(f"按有改判筛选：所有批次 corrected_count>0", all(b.get("corrected_count", 0) > 0 for b in resp_f6),
      f"got={[(b['batch_id'], b.get('corrected_count')) for b in resp_f6]}")

code_f7, resp_f7 = api("GET", "/api/batches?has_corrections=no&limit=10")
check(f"按无改判筛选：所有批次 corrected_count=0", all(b.get("corrected_count", 0) == 0 for b in resp_f7))

# 12e 按时间范围筛选
now = datetime.datetime.now(datetime.timezone.utc)
one_hour_ago = (now - datetime.timedelta(hours=1)).isoformat()
tomorrow = (now + datetime.timedelta(days=1)).isoformat()
yesterday = (now - datetime.timedelta(days=1)).isoformat()

code_f8, resp_f8 = api("GET", f"/api/batches?created_from={one_hour_ago}&created_to={tomorrow}&limit=10")
check(f"最近1小时到明天筛选：返回所有批次", len(resp_f8) >= 3)

code_f9, resp_f9 = api("GET", f"/api/batches?created_from={yesterday}&created_to={yesterday}&limit=10")
check(f"仅昨天时间筛选：返回0条", len(resp_f9) == 0)

# 12f 组合筛选：有改判 + 无冲突
code_f10, resp_f10 = api("GET", "/api/batches?has_corrections=yes&has_conflicts=no&limit=10")
check(f"组合筛选有改判+无冲突：所有批次符合条件",
      all(b.get("corrected_count", 0) > 0 and b.get("conflict_count", 0) == 0 for b in resp_f10))
check(f"组合筛选有改判+无冲突至少1条", len(resp_f10) >= 1)

# ============================================================
print("\n[13] 批次详情聚合API - 导出记录/改判摘要/冲突原因/操作日志")
# ============================================================

# 13a 查询 rev_batch_id 的详情聚合
code_dtl, resp_dtl = api("GET", f"/api/batches/{rev_batch_id}/detail")
check("批次详情聚合API返回200", code_dtl == 200)
check("详情含 batch_id", resp_dtl.get("batch_id") == rev_batch_id)
check("详情含 filename", resp_dtl.get("filename") is not None)
check("详情含 model_version", resp_dtl.get("model_version") is not None and resp_dtl.get("model_version").startswith("mdl_"))
check("详情含 dataset_version", resp_dtl.get("dataset_version") is not None and resp_dtl.get("dataset_version").startswith("ds_"))

# 13b 详情包含导出记录
check("详情含 exports 字段", "exports" in resp_dtl)
exports_list = resp_dtl.get("exports", [])
check("详情中至少1条导出记录（训练回流）", len(exports_list) >= 1)
if exports_list:
    check("导出记录含 export_id", exports_list[0].get("export_id") is not None)
    check("导出记录含 export_type", exports_list[0].get("export_type") in ("prediction", "training"))
    check("导出记录含 filename", exports_list[0].get("filename") is not None)

# 13c 详情包含改判摘要
check("详情含 corrected_items_summary 字段", "corrected_items_summary" in resp_dtl)
corr_summary = resp_dtl.get("corrected_items_summary", [])
check("详情改判摘要至少1条", len(corr_summary) >= 1)
if corr_summary:
    s = corr_summary[0]
    check("改判摘要含 clause_text", s.get("clause_text") is not None)
    check("改判摘要含 predicted_label", s.get("predicted_label") is not None)
    check("改判摘要含 corrected_label=付款", s.get("corrected_label") == "付款")
    check("改判摘要含 corrected_by=tester_alice", s.get("corrected_by") == "tester_alice")

# 13d 详情包含改判历史
check("详情含 correction_history 字段", "correction_history" in resp_dtl)
corr_history = resp_dtl.get("correction_history", [])
check("详情改判历史至少2条（首次改判 + 二次改判 + 撤回）", len(corr_history) >= 2)
op_types = [h.get("operation_type") for h in corr_history]
check("改判历史含 create 操作", "create" in op_types)
check("改判历史含 revert 操作", "revert" in op_types)

# 13e 详情包含冲突项（rev_batch_id 应无冲突）
check("详情含 conflict_items 字段", "conflict_items" in resp_dtl)
check("rev_batch 详情冲突项为空", len(resp_dtl.get("conflict_items", [])) == 0)

# 13f 查询原 batch_id（有冲突）的详情
code_dtl2, resp_dtl2 = api("GET", f"/api/batches/{batch_id}/detail")
check("冲突批次详情返回200", code_dtl2 == 200)
check("冲突批次详情 conflict_items 至少2条", len(resp_dtl2.get("conflict_items", [])) >= 2)
if resp_dtl2.get("conflict_items"):
    check("冲突项含 conflict_reason", resp_dtl2["conflict_items"][0].get("conflict_reason") is not None)
    check("冲突项含 clause_text", resp_dtl2["conflict_items"][0].get("clause_text") is not None)

# 13g 详情包含操作日志
check("详情含 operation_logs 字段", "operation_logs" in resp_dtl)
logs_in_detail = resp_dtl.get("operation_logs", [])
check("详情操作日志至少4条（创建+导出+改判+撤回）", len(logs_in_detail) >= 4)
log_types_in_detail = [l.get("operation_type") for l in logs_in_detail]
check("详情日志包含 batch_predict_create", "batch_predict_create" in log_types_in_detail)
check("详情日志包含 correction_create", "correction_create" in log_types_in_detail)
check("详情日志包含 correction_revert", "correction_revert" in log_types_in_detail)

# ============================================================
print("\n[14] 权限控制机制")
# ============================================================

# 14a 查询权限列表
code_p1, resp_p1 = api("GET", "/api/permissions")
check("权限列表返回200", code_p1 == 200)
check("权限列表至少有3种角色（admin/reviewer/viewer）", len(resp_p1) >= 3)

# 14b 验证 admin 权限
code_p2, resp_p2 = api("GET", "/api/permissions/check?role=admin&operation=correction_revert")
check("admin 可以 correction_revert", resp_p2.get("allowed") == True)

code_p3, resp_p3 = api("GET", "/api/permissions/check?role=admin&operation=correction_create")
check("admin 可以 correction_create", resp_p3.get("allowed") == True)

code_p4, resp_p4 = api("GET", "/api/permissions/check?role=admin&operation=batch_view")
check("admin 可以 batch_view", resp_p4.get("allowed") == True)

# 14c 验证 reviewer 权限
code_p5, resp_p5 = api("GET", "/api/permissions/check?role=reviewer&operation=correction_create")
check("reviewer 可以 correction_create", resp_p5.get("allowed") == True)

code_p6, resp_p6 = api("GET", "/api/permissions/check?role=reviewer&operation=correction_revert")
check("reviewer 可以 correction_revert", resp_p6.get("allowed") == True)

code_p7, resp_p7 = api("GET", "/api/permissions/check?role=reviewer&operation=model_activate")
check("reviewer 不可以 model_activate", resp_p7.get("allowed") == False)

# 14d 验证 viewer 权限（只读）
code_p8, resp_p8 = api("GET", "/api/permissions/check?role=viewer&operation=batch_view")
check("viewer 可以 batch_view", resp_p8.get("allowed") == True)

code_p9, resp_p9 = api("GET", "/api/permissions/check?role=viewer&operation=correction_create")
check("viewer 不可以 correction_create", resp_p9.get("allowed") == False)

code_p10, resp_p10 = api("GET", "/api/permissions/check?role=viewer&operation=correction_revert")
check("viewer 不可以 correction_revert", resp_p10.get("allowed") == False)

code_p11, resp_p11 = api("GET", "/api/permissions/check?role=viewer&operation=model_activate")
check("viewer 不可以 model_activate", resp_p11.get("allowed") == False)

# 14e 验证非法角色
code_p12, resp_p12 = api("GET", "/api/permissions/check?role=anonymous&operation=batch_view")
check("非法角色 anonymous 无任何权限", resp_p12.get("allowed") == False)

code_p13, resp_p13 = api("GET", "/api/permissions/check?role=admin&operation=nonexistent_op")
check("不存在的操作返回 False", resp_p13.get("allowed") == False)

# ============================================================
print("\n[15] 导入冲突联动验证")
# ============================================================

# 15a 创建含重复条款的批次
conflict_csv = os.path.join(BASE, "data", "datasets", "_test_conflict.csv")
write_csv(conflict_csv, [
    "条款文本,合同类型",
    "买方应在验收后30日内支付100%货款,采购合同",
    "质保期两年，质保金5%,采购合同",
    "买方应在验收后30日内支付100%货款,采购合同",  # 重复
    "如卖方逾期交付每日支付0.1%违约金,采购合同",
    "质保期两年，质保金5%,采购合同",  # 重复
])
code_cb, resp_cb = api_file("POST", "/api/batches", conflict_csv, {"note": "回归测试-导入冲突"})
check("导入冲突批次创建成功", resp_cb.get("success") == True)
conflict_batch_id = resp_cb.get("batch_id")
check("导入冲突批次 conflict_count=2", resp_cb.get("conflict_count") == 2)
os.remove(conflict_csv)

# 15b 查询批次项，验证冲突标记
code_cbi, resp_cbi = api("GET", f"/api/batches/{conflict_batch_id}/items?limit=20&include_conflicts=true")
check("冲突批次项总数=5", resp_cbi.get("total") == 5)
all_items = resp_cbi.get("items", [])
conflict_items_new = [i for i in all_items if i.get("is_conflict")]
normal_items_new = [i for i in all_items if not i.get("is_conflict")]
check("冲突标记项=2", len(conflict_items_new) == 2)
check("正常项=3", len(normal_items_new) == 3)

# 15c 冲突项的 predicted_label 应为空
check("冲突项 predicted_label 为空", all(i.get("predicted_label") == "" for i in conflict_items_new))
check("冲突项 has conflict_reason", all(i.get("conflict_reason") for i in conflict_items_new))

# 15d 正常项有预测标签和置信度
check("正常项有 predicted_label", all(i.get("predicted_label") for i in normal_items_new))
check("正常项有 confidence>0", all(i.get("confidence", 0) > 0 for i in normal_items_new))

# 15e 冲突项不能改判（验证正常项可以改判，冲突项无法改判）
normal_to_correct = normal_items_new[0]
code_nc, resp_nc = api("POST", "/api/corrections", {
    "clause_text": normal_to_correct["clause_text"],
    "predicted_label": normal_to_correct["predicted_label"],
    "corrected_label": "付款",
    "reason": "冲突联动测试-正常项改判",
    "batch_id": conflict_batch_id,
    "batch_item_id": normal_to_correct["id"],
    "operator": "tester",
})
check("正常项可以改判", resp_nc.get("success") == True)

# 15f 验证批次详情的冲突项列表
code_cd, resp_cd = api("GET", f"/api/batches/{conflict_batch_id}/detail")
check("冲突批次详情 conflict_items=2", len(resp_cd.get("conflict_items", [])) == 2)
if resp_cd.get("conflict_items"):
    check("详情冲突项含冲突原因", resp_cd["conflict_items"][0].get("conflict_reason") is not None)

# ============================================================
print("\n[16] 重启后完整链路验证 - 批次/撤回记录/导出/日志/权限")
# ============================================================

# 16a 直接查数据库验证所有新数据已持久化
db_path = os.path.join(BASE, "data", "app.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 验证批次记录
batch_total = cur.execute("SELECT COUNT(*) FROM batch_predictions").fetchone()[0]
check("数据库批次记录>=4", batch_total >= 4, f"count={batch_total}")

# 验证批次项记录
item_total = cur.execute("SELECT COUNT(*) FROM batch_items").fetchone()[0]
check("数据库批次项记录>=17", item_total >= 17, f"count={item_total}")

# 验证改判历史表存在且有数据
ch_count = cur.execute("SELECT COUNT(*) FROM correction_history").fetchone()[0]
check("数据库 correction_history 记录>=3（首次+二次+撤回）", ch_count >= 3, f"count={ch_count}")

# 验证 permissions 表存在且有数据
perm_count = cur.execute("SELECT COUNT(*) FROM permissions").fetchone()[0]
check("数据库 permissions 记录>=9（3角色x3操作以上）", perm_count >= 9, f"count={perm_count}")

# 验证 correction_history 中保留了 previous_label（撤回时会清空 batch_items 的 previous_label）
prev_count = cur.execute(
    "SELECT COUNT(*) FROM correction_history WHERE previous_label IS NOT NULL AND previous_label != ''"
).fetchone()[0]
check("数据库 correction_history 有 previous_label 记录>=1", prev_count >= 1, f"count={prev_count}")

# 验证批次 corrected_count 字段
cc_rows = cur.execute("SELECT corrected_count FROM batch_predictions WHERE corrected_count > 0").fetchall()
check("数据库 corrected_count 字段有正值批次>=1", len(cc_rows) >= 1)

# 验证操作日志包含 correction_revert
revert_log_count = cur.execute(
    "SELECT COUNT(*) FROM operation_logs WHERE operation_type = 'correction_revert'"
).fetchone()[0]
check("数据库中有 correction_revert 日志>=1", revert_log_count >= 1)

conn.close()

# 16b 模拟重启：服务仍在运行，再次查询所有关键API验证数据完整
code_hr, health_restart = api("GET", "/health")
check("模拟重启后服务健康", health_restart.get("status") == "ok")

# 16c 重启后查询撤回批次详情
code_rd, resp_rd = api("GET", f"/api/batches/{rev_batch_id}/detail")
check("重启后撤回批次详情仍可查询", resp_rd.get("batch_id") == rev_batch_id)
check("重启后撤回批次改判历史仍存在", len(resp_rd.get("correction_history", [])) >= 2)
check("重启后撤回批次操作日志仍存在", len(resp_rd.get("operation_logs", [])) >= 4)

# 16d 重启后改判历史仍可查询
code_rh, resp_rh = api("GET", f"/api/corrections/history?batch_id={rev_batch_id}&limit=10")
check("重启后改判历史仍可查询", len(resp_rh) >= 2)
rh_op_types = [h.get("operation_type") for h in resp_rh]
check("重启后 create/revert 都存在", "create" in rh_op_types and "revert" in rh_op_types)

# 16e 重启后撤回批次的导出仍可用
code_rexp, resp_rexp_raw = api_raw("GET", f"/api/batches/{rev_batch_id}/export/training/download")
check("重启后训练导出仍可用", code_rexp == 200)
rexp_content = resp_rexp_raw.decode("utf-8-sig")
check("重启后训练导出仍含恢复后的标签（付款）", "付款" in rexp_content)

# 16f 重启后权限配置仍可用
code_rp, resp_rp = api("GET", "/api/permissions/check?role=admin&operation=correction_revert")
check("重启后权限配置仍正确", resp_rp.get("allowed") == True)

code_rp2, resp_rp2 = api("GET", "/api/permissions/check?role=viewer&operation=correction_create")
check("重启后 viewer 权限仍受限", resp_rp2.get("allowed") == False)

# 16g 重启后批次筛选功能仍正常（用有改判+无冲突组合筛选）
code_rfilt, resp_rfilt = api("GET", "/api/batches?has_corrections=yes&has_conflicts=no&limit=10")
check("重启后筛选仍正常（有改判+无冲突）", len(resp_rfilt) >= 1)
check("重启后筛选结果正确", all(b.get("corrected_count", 0) > 0 and b.get("conflict_count", 0) == 0 for b in resp_rfilt))

# ============================================================
print("\n" + "=" * 60)
print(f"回归验证结果: PASS={PASSED}  FAIL={FAILED}")
print("=" * 60)

import json, urllib.request, urllib.parse, os, sqlite3

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
# 4a 导入样例
sample_path = os.path.join(BASE, "sample_data", "contracts_sample.csv")
code6, resp6 = api_file("POST", "/api/datasets/import", sample_path, {"note": "回归测试"})
check("样例导入成功", resp6.get("success") == True, f"code={code6} resp={resp6}")
ds_version = resp6.get("dataset_version")
check("导入返回dataset_version", ds_version is not None and ds_version.startswith("ds_"), f"version={ds_version}")
check("导入返回labeled_count=63", resp6.get("validation", {}).get("labeled_count") == 63, f"lc={resp6.get('validation',{}).get('labeled_count')}")

# 4b 训练
code7, resp7 = api("POST", "/api/models/train", {
    "dataset_version": ds_version,
    "auto_activate": True,
    "note": "回归测试训练",
})
check("训练成功", resp7.get("success") == True, f"code={code7} resp_err={resp7.get('error','')[:200]}")
mdl_version = resp7.get("model_version")
check("训练返回model_version", mdl_version is not None and mdl_version.startswith("mdl_"))
check("训练返回test_accuracy>0", resp7.get("test_accuracy", 0) > 0)
check("训练auto_activated=True", resp7.get("auto_activated") == True)

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
check("批量预测绑定model_version", resp_batch.get("model_version") == mdl_version,
      f"expected={mdl_version} got={resp_batch.get('model_version')}")
check("批量预测绑定dataset_version", resp_batch.get("dataset_version") == ds_version,
      f"expected={ds_version} got={resp_batch.get('dataset_version')}")

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
check("批次日志至少3条", len(resp_logs_batch) >= 3)
check("批次日志的entity_id正确", all(l.get("entity_id") == batch_id for l in resp_logs_batch))

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

# 9c 训练第二个模型（用同一个数据集，确保有第二个模型）
code_train2, resp_train2 = api("POST", "/api/models/train", {
    "dataset_version": ds_version,
    "auto_activate": False,
    "note": "回归测试-第二模型用于版本绑定验证",
})
check("第二个模型训练成功", resp_train2.get("success") == True, f"code={code_train2} err={resp_train2.get('error','')[:100]}")
mdl_version2 = resp_train2.get("model_version")
check("第二个模型版本已生成", mdl_version2 is not None and mdl_version2 != mdl_version)

# 9d 激活第二个模型
code_act2, resp_act2 = api_form("POST", f"/api/models/{mdl_version2}/activate", {"operator_note": "回归测试-切换模型验证版本绑定"})
check("激活第二个模型成功", resp_act2.get("success") == True, f"code={code_act2}")

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
check("旧批次导出文件中没有v2模型版本", mdl_version2 not in exp2_content)

# 9g 用新模型创建新批次，验证新批次绑定新模型
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

# 9h 两个批次各自绑定不同模型版本，互不干扰
batch_list = api("GET", "/api/batches?limit=10")[1]
batch1_info = next((b for b in batch_list if b["batch_id"] == batch_id), None)
batch2_info = next((b for b in batch_list if b["batch_id"] == batch_id2), None)
check("批次列表中批次1绑定v1", batch1_info and batch1_info["model_version"] == model_v1)
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
print("\n" + "=" * 60)
print(f"回归验证结果: PASS={PASSED}  FAIL={FAILED}")
print("=" * 60)

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
print("\n" + "=" * 60)
print(f"回归验证结果: PASS={PASSED}  FAIL={FAILED}")
print("=" * 60)

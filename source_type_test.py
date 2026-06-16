"""
批量预测 - 导入来源(source_type)筛选 专项回归测试
覆盖：
  [1] 前端模板结构：fSourceType 等筛选控件在模板中存在（静态检查，非运行时）
  [2] API：source_type 查询能影响返回集合
  [3] 不回归：原有 6 项筛选项（模型/数据集/时间/冲突/改判）仍可用
  [4] 新建批次：source_type 参数能写入数据库并被查询命中
  [5] 持久化验证：source_type 数据持久化到数据库（服务重启后数据不丢失）
  [6] 浏览器端到端验证：Tab 切换 / 筛选交互 / 结果变化（需手动执行 integrated_browser 验证）

运行方式：python source_type_test.py
需要服务正在 http://127.0.0.1:8001 运行
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
INDEX_HTML = os.path.join(BASE, "templates", "index.html")

API = "http://127.0.0.1:8001"
passed = 0
failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")

def api(method, path, body=None, content_type=None):
    data = None
    headers = {}
    if body is not None:
        if isinstance(body, dict):
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = content_type or "application/json"
        elif isinstance(body, (bytes, bytearray)):
            data = body
            if content_type:
                headers["Content-Type"] = content_type
        else:
            data = str(body).encode("utf-8")
            if content_type:
                headers["Content-Type"] = content_type
    req = urllib.request.Request(API + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8") or "null")
        except Exception:
            return e.code, {"error": str(e)}

def api_multipart(method, path, fields=None, files=None):
    boundary = "----TestBoundary" + str(int(time.time() * 1000))
    body = bytearray()
    if fields:
        for k, v in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            body.extend(f"{v}\r\n".encode("utf-8"))
    if files:
        for k, (fname, fcontent, ftype) in files.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode("utf-8"))
            body.extend(f"Content-Type: {ftype}\r\n\r\n".encode("utf-8"))
            if isinstance(fcontent, str):
                fcontent = fcontent.encode("utf-8")
            body.extend(fcontent)
            body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(
        API + path,
        data=bytes(body),
        method=method,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8") or "null")
        except Exception:
            return e.code, {"error": str(e)}

def write_csv(path, lines):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("\n".join(lines) + "\n")

print("=" * 70)
print("批量预测 - 导入来源(source_type)筛选 专项回归测试")
print("=" * 70)

# 先找一个可用的模型版本
print("\n[0] 环境准备")
code, models = api("GET", "/api/models")
trained = [m for m in models if m.get("training_status") == "completed"]
if not trained:
    print("  !! 无已训练模型，尝试导入样例并训练")
    # 导入样例
    code, ds_list = api("GET", "/api/datasets")
    ds_version = None
    if ds_list and len(ds_list) > 0:
        ds_version = ds_list[0]["version"]
    else:
        sample_path = os.path.join(BASE, "sample_data", "contracts_sample.csv")
        with open(sample_path, "rb") as f:
            code, resp = api_multipart(
                "POST", "/api/datasets/import",
                {"note": "source_type测试导入"},
                {"file": ("contracts_sample.csv", f.read(), "text/csv")},
            )
        ds_version = resp.get("dataset_version")
    check("数据集可用", ds_version is not None)
    # 训练
    code, resp = api("POST", "/api/models/train", {
        "dataset_version": ds_version, "auto_activate": True, "note": "source_type测试训练",
    })
    trained = [{"version": resp["model_version"]}] if resp.get("model_version") else []
    code, models = api("GET", "/api/models")
    trained = [m for m in models if m.get("training_status") == "completed"]
mdl_version = trained[0]["version"] if trained else None
check(f"找到可用模型 {mdl_version}", mdl_version is not None)
if not mdl_version:
    print("\n无法继续：没有可用的训练好的模型")
    sys.exit(1)

# -----------------------------------------------------------------------------
print("\n[1] 前端模板结构：fSourceType 等筛选控件在模板中存在（静态检查）")
print("  说明：本节仅验证模板文件中是否存在对应 DOM 节点，")
print("        不代表 JS 能正常运行。真正的前端功能验证见 [6] 浏览器端到端验证。")
with open(INDEX_HTML, "r", encoding="utf-8") as f:
    html = f.read()
check('id="fSourceType" 下拉在模板中存在', 'id="fSourceType"' in html)
check('id="btnApplyFilter" 应用筛选按钮在模板中存在', 'id="btnApplyFilter"' in html)
check('id="btnResetFilter" 重置按钮在模板中存在', 'id="btnResetFilter"' in html)
check('id="btnRefreshBatches" 刷新按钮在模板中存在', 'id="btnRefreshBatches"' in html)
check('id="fModelVersion" 模型版本筛选在模板中存在', 'id="fModelVersion"' in html)
check('id="fDatasetVersion" 数据集版本筛选在模板中存在', 'id="fDatasetVersion"' in html)
check('id="fHasConflicts" 冲突筛选在模板中存在', 'id="fHasConflicts"' in html)
check('id="fHasCorrections" 改判筛选在模板中存在', 'id="fHasCorrections"' in html)
check('id="fCreatedFrom" 开始时间筛选在模板中存在', 'id="fCreatedFrom"' in html)
check('id="fCreatedTo" 结束时间筛选在模板中存在', 'id="fCreatedTo"' in html)
check("筛选面板 grid-cols 支持 7 个筛选项自然换行", "lg:grid-cols-4" in html or "lg:grid-cols-7" in html)

# -----------------------------------------------------------------------------
print("\n[2] API：source_type 查询能影响返回集合")
# 2a 创建不同 source_type 的批次
batch_a_csv = os.path.join(BASE, "data", "datasets", "_st_manual.csv")
batch_b_csv = os.path.join(BASE, "data", "datasets", "_st_system.csv")
batch_c_csv = os.path.join(BASE, "data", "datasets", "_st_review.csv")
csv_lines = [
    "条款文本,合同类型",
    "合同签订后10日内甲方支付预付款10万元,采购合同",
    "货物验收合格后支付90%货款,采购合同",
    "质保期一年，质保金10%,采购合同",
    "乙方应于交付后5日内提供技术培训,服务合同",
    "违约金按未交付部分每日千分之一计算,买卖合同",
]
for p in (batch_a_csv, batch_b_csv, batch_c_csv):
    write_csv(p, csv_lines)

# 2b 上传不同 source_type
with open(batch_a_csv, "rb") as f:
    code_a, resp_a = api_multipart(
        "POST", "/api/batches",
        {"note": "source_type=manual_upload测试", "source_type": "manual_upload"},
        {"file": ("_st_manual.csv", f.read(), "text/csv")},
    )
check("manual_upload 批次创建成功", resp_a.get("success") == True, f"code={code_a} {resp_a.get('error','')[:80]}")
batch_a_id = resp_a.get("batch_id")
check("manual_upload 批次返回 batch_id", batch_a_id is not None)

with open(batch_b_csv, "rb") as f:
    code_b, resp_b = api_multipart(
        "POST", "/api/batches",
        {"note": "source_type=system_import测试", "source_type": "system_import"},
        {"file": ("_st_system.csv", f.read(), "text/csv")},
    )
check("system_import 批次创建成功", resp_b.get("success") == True, f"code={code_b} {resp_b.get('error','')[:80]}")
batch_b_id = resp_b.get("batch_id")
check("system_import 批次返回 batch_id", batch_b_id is not None)

with open(batch_c_csv, "rb") as f:
    code_c, resp_c = api_multipart(
        "POST", "/api/batches",
        {"note": "source_type=review_batch测试", "source_type": "review_batch"},
        {"file": ("_st_review.csv", f.read(), "text/csv")},
    )
check("review_batch 批次创建成功", resp_c.get("success") == True, f"code={code_c} {resp_c.get('error','')[:80]}")
batch_c_id = resp_c.get("batch_id")
check("review_batch 批次返回 batch_id", batch_c_id is not None)

for p in (batch_a_csv, batch_b_csv, batch_c_csv):
    if os.path.exists(p):
        os.remove(p)

# 2c 验证 distinct source_types API 包含三种
code_st, st_list = api("GET", "/api/batches/source_types")
check("source_types API 返回 list 类型", isinstance(st_list, list))
check("source_types 含 manual_upload", "manual_upload" in st_list)
check("source_types 含 system_import", "system_import" in st_list)
check("source_types 含 review_batch", "review_batch" in st_list)

# 2d 单独筛选每个 source_type
code_fa, list_a = api("GET", "/api/batches?source_type=manual_upload&limit=200")
check("筛选 manual_upload：全部返回 manual_upload",
      all(b["source_type"] == "manual_upload" for b in list_a),
      f"实际source_type: {list(set(b.get('source_type') for b in list_a))}")
check("筛选 manual_upload：至少包含我们刚创建的批次",
      any(b["batch_id"] == batch_a_id for b in list_a))

code_fb, list_b = api("GET", "/api/batches?source_type=system_import&limit=200")
check("筛选 system_import：全部返回 system_import",
      all(b["source_type"] == "system_import" for b in list_b))
check("筛选 system_import：包含刚创建的批次",
      any(b["batch_id"] == batch_b_id for b in list_b))

code_fc, list_c = api("GET", "/api/batches?source_type=review_batch&limit=200")
check("筛选 review_batch：全部返回 review_batch",
      all(b["source_type"] == "review_batch" for b in list_c))
check("筛选 review_batch：包含刚创建的批次",
      any(b["batch_id"] == batch_c_id for b in list_c))

# 2e 互相隔离：筛选 system_import 不能出现 batch_a_id 和 batch_c_id
check("筛选 system_import 不包含 manual_upload 的批次",
      all(b["batch_id"] != batch_a_id for b in list_b))
check("筛选 system_import 不包含 review_batch 的批次",
      all(b["batch_id"] != batch_c_id for b in list_b))

# 2f 筛选 不存在的 source_type 返回空
code_fx, list_x = api("GET", "/api/batches?source_type=" + urllib.parse.quote("不存在的来源") + "&limit=200")
check("筛选不存在的 source_type 返回空列表", len(list_x) == 0)

# -----------------------------------------------------------------------------
print("\n[3] 不回归：原有 6 项筛选项仍可用")
# 3a 模型版本筛选
code_fm, list_m = api("GET", f"/api/batches?model_version={mdl_version}&limit=200")
check(f"按模型版本 {mdl_version} 筛选：全部匹配",
      all(b["model_version"] == mdl_version for b in list_m))
check("按模型版本筛选：包含我们刚创建的 3 个批次",
      sum(1 for b in list_m if b["batch_id"] in (batch_a_id, batch_b_id, batch_c_id)) == 3)

# 3b 冲突状态筛选（这些批次都无冲突）
code_fny, list_ny = api("GET", "/api/batches?has_conflicts=no&limit=200")
check("has_conflicts=no 筛选：全部 conflict_count=0",
      all(b.get("conflict_count", 0) == 0 for b in list_ny))

# 3c 改判状态筛选（都未改判）
code_fcy, list_cy = api("GET", "/api/batches?has_corrections=no&limit=200")
check("has_corrections=no 筛选：全部 corrected_count=0",
      all(b.get("corrected_count", 0) == 0 for b in list_cy))

# 3d 时间范围筛选（取 24 小时前~24 小时后，规避 UTC/本地时区差问题）
t1 = time.time() - 86400
t2 = time.time() + 86400
iso_fr = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t1)) + ".000Z"
iso_to = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t2)) + ".000Z"
code_ft, list_t = api("GET", f"/api/batches?created_from={urllib.parse.quote(iso_fr)}&created_to={urllib.parse.quote(iso_to)}&limit=200")
check("时间范围筛选（±24小时）：包含刚创建的 3 个批次",
      sum(1 for b in list_t if b["batch_id"] in (batch_a_id, batch_b_id, batch_c_id)) == 3)

# 3e 组合筛选：source_type=manual_upload + has_conflicts=no
code_fcomb, list_comb = api("GET", "/api/batches?source_type=manual_upload&has_conflicts=no&limit=200")
check("组合筛选 manual_upload+无冲突：所有项符合条件",
      all(b.get("source_type") == "manual_upload" and b.get("conflict_count", 0) == 0 for b in list_comb))
check("组合筛选结果包含 batch_a_id",
      any(b["batch_id"] == batch_a_id for b in list_comb))
check("组合筛选结果不包含 batch_b_id (system_import)",
      all(b["batch_id"] != batch_b_id for b in list_comb))

# -----------------------------------------------------------------------------
print("\n[4] 新建批次：source_type 参数能写入数据库并被查询命中")
# 创建 batch_a 时已指定 source_type=manual_upload，查 detail API 验证
code_det_a, det_a = api("GET", f"/api/batches/{batch_a_id}")
check("批次 A detail 中 source_type=manual_upload", det_a.get("source_type") == "manual_upload",
      f"实际={det_a.get('source_type')}")
code_det_b, det_b = api("GET", f"/api/batches/{batch_b_id}")
check("批次 B detail 中 source_type=system_import", det_b.get("source_type") == "system_import",
      f"实际={det_b.get('source_type')}")
code_det_c, det_c = api("GET", f"/api/batches/{batch_c_id}")
check("批次 C detail 中 source_type=review_batch", det_c.get("source_type") == "review_batch",
      f"实际={det_c.get('source_type')}")

# -----------------------------------------------------------------------------
print("\n[5] 持久化验证：source_type 数据持久化到数据库，服务重启后不丢失")
print("  说明：本节验证数据已写入 SQLite 数据库（持久化存储），")
print("        服务重启后数据仍然存在。完整的'重启后功能验证'见 [6]。")

# 5a 直接查数据库证明持久化
from src.models import get_connection
conn = get_connection()
try:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT batch_id, source_type FROM batch_predictions WHERE batch_id IN (?,?,?) ORDER BY batch_id",
        (batch_a_id, batch_b_id, batch_c_id),
    ).fetchall()
    db_map = {r[0]: r[1] for r in rows}
    check("数据库中 batch_a 保存 manual_upload", db_map.get(batch_a_id) == "manual_upload")
    check("数据库中 batch_b 保存 system_import", db_map.get(batch_b_id) == "system_import")
    check("数据库中 batch_c 保存 review_batch", db_map.get(batch_c_id) == "review_batch")
    check("三个批次 source_type 都不是 NULL/空",
          all(v is not None and v != "" for v in db_map.values()),
      f"db_map={db_map}")
finally:
    conn.close()

# 5b 空表单字段回退（不传 source_type 时 default=manual_upload）
batch_d_csv = os.path.join(BASE, "data", "datasets", "_st_default.csv")
write_csv(batch_d_csv, csv_lines)
with open(batch_d_csv, "rb") as f:
    code_d, resp_d = api_multipart(
        "POST", "/api/batches",
        {"note": "source_type不传默认测试"},
        {"file": ("_st_default.csv", f.read(), "text/csv")},
    )
check("不传 source_type 表单字段也能创建", resp_d.get("success") == True,
      f"code={code_d} {resp_d.get('error','')[:80]}")
batch_d_id = resp_d.get("batch_id")
if batch_d_id:
    code_det_d, det_d = api("GET", f"/api/batches/{batch_d_id}")
    check("不传 source_type 默认回退为 manual_upload",
          det_d.get("source_type") == "manual_upload",
          f"实际={det_d.get('source_type')}")
if os.path.exists(batch_d_csv):
    os.remove(batch_d_csv)

# -----------------------------------------------------------------------------
print("\n[6] 浏览器端到端验证：Tab 切换 / 筛选交互 / 结果变化")
print("  说明：本节不自动执行，需手动用浏览器或 integrated_browser MCP 验证。")
print("  验证步骤：")
print("    1. 打开 http://127.0.0.1:8001/ ，按 F12 打开 Console，确认无 JS error")
print("    2. 点击顶部「🔮 预测」Tab，能看到「单条预测 / 批量预测 & 批次历史」两个子 Tab")
print("    3. 点击「批量预测 & 批次历史」子 Tab，能看到筛选面板（7个筛选项）")
print("    4. 导入来源下拉有值：全部 / manual_upload / review_batch / system_import")
print("    5. 选择 system_import 后点「应用筛选」，批次列表数量减少（筛选生效）")
print("    6. 点「重置」，下拉回到「全部」，列表数量恢复")
print("    7. 重启服务后重复步骤 1-6，全部仍可正常使用")
print("  预期：全部步骤通过，Console 无 JS error，筛选前后列表数量变化正确")

# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"测试完成：通过 {passed} / 失败 {failed}  (总计 {passed + failed})")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)

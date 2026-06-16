import json, urllib.request, urllib.parse, os, io

API = "http://127.0.0.1:8001"
BASE = os.path.dirname(os.path.abspath(__file__))

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
            b = e.read().decode("utf-8")
            try:
                return e.code, json.loads(b)
            except:
                return e.code, {"raw": b[:500]}
        except Exception as ex:
            return e.code, {"error": str(ex)}

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
            b = e.read().decode("utf-8")
            try:
                return e.code, json.loads(b)
            except:
                return e.code, {"raw": b[:500]}
        except Exception as ex:
            return e.code, {"error": str(ex)}

# 1. 健康检查
print("=== 健康检查 ===")
c, r = api("GET", "/health")
print(c, r)

# 2. 获取模型列表
print("\n=== 模型列表 ===")
c, r = api("GET", "/api/models")
print(c, r)

# 3. 获取数据集列表
print("\n=== 数据集列表 ===")
c, r = api("GET", "/api/datasets")
print(c, r[:2] if isinstance(r, list) else r)

# 4. 获取现役模型
print("\n=== 现役模型 ===")
c, r = api("GET", "/api/models/active")
print(c, r)

# 5. 测试导入数据集（样例CSV）
print("\n=== 测试导入样例数据集 ===")
sample_path = os.path.join(BASE, "sample_data", "contracts_sample.csv")
c, r = api_file("POST", "/api/datasets/import", sample_path, {"note": "debug test"})
print(c, json.dumps(r, ensure_ascii=False)[:500])

# 6. 测试训练API - 如果有数据集的话
print("\n=== 测试训练API ===")
c, ds_list = api("GET", "/api/datasets")
if isinstance(ds_list, list) and ds_list:
    dv = ds_list[0]["version"]
    print(f"使用数据集: {dv}")
    c, r = api("POST", "/api/models/train", {
        "dataset_version": dv,
        "auto_activate": False,
        "note": "debug train",
    })
    print(c, json.dumps(r, ensure_ascii=False)[:500])
    
    if r.get("success") and r.get("model_version"):
        mv = r["model_version"]
        print(f"\n=== 测试激活API - 激活 {mv} ===")
        c, r = api_form("POST", f"/api/models/{mv}/activate", {"operator_note": "debug activate"})
        print(c, json.dumps(r, ensure_ascii=False)[:500])

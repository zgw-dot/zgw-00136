# 合同条款风险标注训练工具（本地版）

**纯本地运行，不调用任何外部大模型或 SaaS。** 支持导入带标签 CSV → 训练轻量 TF-IDF + 逻辑回归分类器 → 评估（带完整版本追溯）→ 预测 → 人工改判（记录原因）→ 模型/数据集版本化管理 + 回滚。

---

## 1. 功能速览

| 模块 | 说明 |
|------|------|
| 数据导入 | 上传 CSV，校验必填列（`条款文本` `风险标签`），**空标签行直接拒绝导入**，生成 **数据集版本** |
| 模型训练 | TF-IDF 中文分词 + LogisticRegression（lbfgs, balanced class_weight），训练失败**不替换现役模型** |
| 评估报告 | Accuracy / Precision / Recall / F1（macro）+ 混淆矩阵 + 分类报告；**每项指标绑定模型+数据集版本** |
| 条款预测 | 单条 / 批量 CSV 预测，返回 Top-K 概率；未训练直接预测返回明确错误 |
| 人工改判 | 保存条款原文 + 预测标签 + 改判标签 + **改判原因** + 归属模型版本 |
| 版本/回滚 | 模型版本激活/回滚；所有切换写入 `rollback_logs`；**重启后状态一致**（SQLite + 文件持久化）|
| 数据持久化 | SQLite（`data/app.db`） + joblib 模型文件 + CSV 数据集副本 + JSON 评估报告 |

风险标签固定为 3 类：**`付款` / `违约` / `自动续约`**。

---

## 2. 复现实验（三步启动）

### 环境要求
- Python 3.10+（已验证 3.13）
- Windows / Linux / macOS 均可

### Step 1. 安装依赖

```bash
cd d:\workSpace\AI__SPACE\02-label\zgw-00136
pip install -r requirements.txt
```

依赖清单（全部本地、无远程 API）：
- `fastapi` + `uvicorn`：Web 服务
- `scikit-learn`：TF-IDF + LogisticRegression + 指标
- `pandas` / `numpy`：数据处理
- `joblib`：模型序列化
- `jinja2`：前端模板

### Step 2. 启动服务

```bash
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

浏览器打开 **http://127.0.0.1:8001**

### Step 3. 完整实验流程（CLI + curl 或直接用网页）

#### 3.1 健康检查
```bash
curl http://127.0.0.1:8001/health
```

#### 3.2 校验内置样例 CSV
样例文件：`sample_data/contracts_sample.csv`（63 条条款，3 类各 21 条，0 条空标签）。

```bash
curl -X POST -F "file=@sample_data/contracts_sample.csv" \
  http://127.0.0.1:8001/api/datasets/validate
```

#### 3.3 导入数据集（自动生成版本号 `ds_*`）

```bash
curl -X POST \
  -F "file=@sample_data/contracts_sample.csv" \
  -F "note=内置样例数据" \
  -F "drop_empty_text=true" \
  http://127.0.0.1:8001/api/datasets/import
```

返回示例：
```json
{
  "success": true,
  "dataset_version": "ds_20260617000000_a1b2c3",
  "validation": { "labeled_count": 63, "label_distribution": {"付款":21,"违约":21,"自动续约":21} }
}
```

#### 3.4 训练模型（自动生成 `mdl_*` 版本号）

把上一步的 `dataset_version` 替换进去：

```bash
curl -X POST http://127.0.0.1:8001/api/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_version": "ds_20260617000000_a1b2c3",
    "hyperparams": {
      "vectorizer": {"max_features": 5000, "ngram_range": [1,2], "min_df": 1},
      "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"},
      "split": {"test_size": 0.2, "random_state": 42}
    },
    "note": "默认超参首次训练",
    "auto_activate": true
  }'
```

> **训练失败保护**：如果数据只有 1 个类别 / 测试准确率 < 0.3 / 模型文件序列化异常，`model_versions.training_status` 会被标记为 `failed`，**不会自动激活，也不会替换现役模型**。

#### 3.5 运行评估（返回绑定模型+数据集版本的完整报告）

```bash
curl -X POST http://127.0.0.1:8001/api/evaluations \
  -H "Content-Type: application/json" \
  -d '{"test_ratio": 0.2, "random_state": 42}'
```

返回的 `traceability` 字段包含：
- `model_version`、`dataset_version`
- `dataset_filename`、`dataset_row_count`、`dataset_labeled_count`
- `model_hyperparams`、`model_feature_count`
- 各类别指标 + 混淆矩阵 + 错分样例

**导出文本报告（人类可读 + 版本追溯）**：
```bash
curl "http://127.0.0.1:8001/api/evaluations/1/report.txt"
```

**导出 JSON 报告**：
```bash
curl "http://127.0.0.1:8001/api/evaluations/1/report.json"
```

#### 3.6 单条条款预测

```bash
curl -X POST http://127.0.0.1:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "clause_text": "如卖方逾期交付超过10日，买方有权解除合同并要求支付合同总金额10%的违约金",
    "contract_type": "采购合同",
    "top_k": 3
  }'
```

**失败路径测试 - 未激活模型时预测**：
1. 删除 `data/app.db` 重启服务；
2. 调用 `/api/predict` 会返回 400 + 错误消息「尚无激活的模型，请先训练并激活一个模型版本」。

#### 3.7 人工改判 + 记录原因

```bash
curl -X POST http://127.0.0.1:8001/api/corrections \
  -H "Content-Type: application/json" \
  -d '{
    "clause_text": "（模型可能判错的某条款原文…）",
    "predicted_label": "付款",
    "corrected_label": "违约",
    "reason": "条款核心在逾期赔偿责任，属于违约类别而非付款时间条款",
    "model_version": "mdl_20260617000000_xxxxxx"
  }'
```

- `corrections` 表强制保存 `reason` 列；
- 每条记录绑定 `model_version`，便于后续增量训练溯源。

#### 3.8 激活 / 回滚模型版本

```bash
curl -X POST http://127.0.0.1:8001/api/models/mdl_xxxxx/activate \
  -F "operator_note=回滚到xx版本，因为新版本在违约类Recall下降5%"
```

所有激活/回滚操作写入 `rollback_logs` 表，页面「版本/回滚」页可完整查看。

---

## 3. 目录结构 & 持久化文件

```
zgw-00136/
├── app.py                      # FastAPI 入口，所有路由
├── requirements.txt
├── sample_data/
│   └── contracts_sample.csv    # 内置样例（63条，3类各21条，0空标签）
├── templates/
│   └── index.html              # 单页前端（导入/训练/评估/预测/改判/版本）
├── src/
│   ├── models.py               # SQLite 初始化 + 表结构
│   ├── version_manager.py      # 数据集/模型版本、回滚日志、评估、改判 CRUD
│   ├── data_manager.py         # CSV 校验 + 导入 + 空标签直接拒绝/必填列/非法标签处理
│   ├── trainer.py              # TF-IDF + Logistic 训练，失败不替换现役
│   ├── predictor.py            # 单条/批量预测 + 改判写入
│   └── evaluator.py            # 评估 + 版本追溯 + 文本/JSON 报告生成
└── data/
    ├── app.db                  # SQLite（重启后模型激活、改判、回滚日志全部一致）
    ├── datasets/               # 导入的数据集副本（版本号命名）
    ├── models/                 # 模型文件（每个版本一个子目录：classifier/vectorizer/label_encoder.joblib）
    ├── evaluations/            # 导出的 JSON 评估报告
    └── corrections/            # 预留目录
```

**重启一致性**：现役模型通过 `model_versions.is_active` 字段标记；改判/回滚/评估历史都由 SQLite 持久化，文件目录副本对应版本号一一匹配，**重启服务后状态与断电前完全一致**。

---

## 4. 失败路径清单（全部已处理）

| 场景 | 处理方式 |
|------|----------|
| CSV 缺少 `条款文本` 或 `风险标签` 列 | 返回 400 + 错误详情 + 校验 JSON |
| CSV 存在空风险标签行 | **导入时直接拒绝**，返回 400 + 空标签行数，不生成数据集版本 |
| 标签不在 `付款/违约/自动续约` | 返回 400 + 非法标签列表，不生成数据集版本 |
| 类别数 < 2 | 训练失败，写入 `failed` 状态 |
| 测试准确率 < 0.3 | 训练失败，不激活 |
| 未激活模型时预测 | 返回 400「尚无激活的模型」 |
| 模型文件损坏/丢失 | 返回 400「模型文件缺失」 |
| 训练中途异常 | 回滚写入 `training_status=failed` + traceback，**不会误替换现役** |
| 人工改判未填原因 | 返回 400「必须填写改判原因」 |

---

## 5. 离线说明

**后端**：代码不包含任何 `requests.post` 外网调用；模型使用 `scikit-learn` 本地训练；所有数据存放在 `data/` 目录下。

**前端**：Tailwind CSS 运行时（`static/tailwind.js`）已下载到本地，页面不引用任何外部 CDN。

**全程完全离线可用，无需联网。**

---

## 6. 命令式快速完整实验（一键跑完所有步骤）

Windows PowerShell：

```powershell
# 1. 安装
pip install -r requirements.txt

# 2. 后台启动
Start-Process uvicorn -ArgumentList "app:app","--host","127.0.0.1","--port","8001" -WindowStyle Hidden
Start-Sleep -Seconds 3

# 3. 导入样例
$importResp = curl.exe -s -X POST `
  -F "file=@sample_data/contracts_sample.csv" `
  -F "note=PowerShell脚本测试" `
  http://127.0.0.1:8001/api/datasets/import | ConvertFrom-Json
$ds = $importResp.dataset_version
Write-Host "数据集版本: $ds"

# 4. 训练
$trainBody = @{ dataset_version=$ds; auto_activate=$true; note="PS自动训练"} | ConvertTo-Json -Depth 8
$trainResp = curl.exe -s -X POST http://127.0.0.1:8001/api/models/train `
  -H "Content-Type: application/json" -d $trainBody | ConvertFrom-Json
Write-Host "模型版本: $($trainResp.model_version) · 测试Acc: $($trainResp.test_accuracy)"

# 5. 评估
$evalResp = curl.exe -s -X POST http://127.0.0.1:8001/api/evaluations `
  -H "Content-Type: application/json" -d "{}" | ConvertFrom-Json
Write-Host "评估ID: $($evalResp.evaluation_id) · F1(macro): $($evalResp.metrics.f1_macro)"

# 6. 导出文本报告
curl.exe -s "http://127.0.0.1:8001/api/evaluations/$($evalResp.evaluation_id)/report.txt"

# 7. 预测
curl.exe -s -X POST http://127.0.0.1:8001/api/predict `
  -H "Content-Type: application/json" `
  -d '{"clause_text":"本合同到期自动续展一年，除非提前30日书面通知终止"}'
```

---

## 7. 数据列约定

```csv
条款文本,风险标签,合同类型,时间
买方应在收到发票后30日内付款,付款,采购合同,2024-01-01
逾期交付超过10日支付10%违约金,违约,采购合同,2024-01-02
期满双方无异议自动续约一年,自动续约,服务合同,2024-01-03
```

- 必填：`条款文本` `风险标签`（**不允许空值，有则导入时拒绝**）
- 可选：`合同类型`（训练时会拼接到特征前增强分类能力）`时间`
- 标签枚举：`付款` `违约` `自动续约`

---

## 8. 常用 API 速查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 前端单页 |
| GET  | `/sample_data/contracts_sample.csv` | 下载内置样例 CSV |
| GET  | `/health` | 健康检查 + 数据统计 |
| GET  | `/api/info` | 允许标签、默认超参、当前现役模型 |
| POST | `/api/datasets/validate` | 校验 CSV |
| POST | `/api/datasets/import` | 导入 CSV 创建数据集版本 |
| GET  | `/api/datasets` | 数据集版本列表 |
| GET  | `/api/datasets/{version}/download` | 下载该版本 CSV |
| POST | `/api/models/train` | 启动训练（auto_activate 可选） |
| GET  | `/api/models` | 模型版本列表 |
| GET  | `/api/models/active` | 现役模型 |
| POST | `/api/models/{version}/activate` | 激活/回滚 + 写日志 |
| POST | `/api/predict` | 单条预测 |
| POST | `/api/predict/batch` | 多条 JSON 预测 |
| POST | `/api/predict/csv` | CSV 批量预测 |
| POST | `/api/corrections` | 记录人工改判（+ 原因） |
| GET  | `/api/corrections` | 改判历史（可按 model_version 过滤） |
| POST | `/api/evaluations` | 运行评估 |
| GET  | `/api/evaluations/{id}` | 评估详情 |
| GET  | `/api/evaluations/{id}/report.txt` | 文本报告 |
| GET  | `/api/evaluations/{id}/report.json` | JSON 报告 |
| GET  | `/api/rollback-logs` | 激活/回滚日志 |

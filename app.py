import os
import tempfile
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.models import init_db, get_connection
from src.data_manager import (
    validate_csv,
    import_csv,
    DataImportError,
    ValidationResult,
    ALLOWED_LABELS,
)
from src.version_manager import (
    list_dataset_versions,
    get_dataset_version,
    list_model_versions,
    get_model_version,
    get_active_model_version,
    activate_model_version,
    list_corrections,
    list_rollback_logs,
    list_evaluations,
    list_operation_logs,
    add_operation_log,
    list_correction_history,
    list_permissions,
    check_permission,
    list_filter_schemes,
    get_filter_scheme,
    get_default_filter_scheme,
    create_filter_scheme,
    update_filter_scheme,
    set_default_filter_scheme,
    delete_filter_scheme,
)
from src.trainer import train_model, TrainingError, DEFAULT_HYPERPARAMS
from src.predictor import (
    predict_single,
    predict_batch,
    predict_csv,
    record_correction,
    revert_correction,
    PredictionError,
)
from src.evaluator import (
    run_evaluation,
    get_evaluation_detail,
    build_summary_text,
    EvaluationError,
    list_evaluations as eval_list,
)
from src.batch_manager import (
    create_batch_prediction,
    list_batches,
    get_batch,
    get_batch_items,
    get_batch_item_count,
    get_batch_detail,
    export_batch_to_csv,
    export_audit_to_csv,
    list_exports,
    get_export,
    BatchError,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

init_db()

app = FastAPI(
    title="合同条款风险标注训练工具",
    description="本地合同条款风险分类器 - 导入CSV → 训练 → 评估 → 预测 → 人工改判",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


class PredictRequest(BaseModel):
    clause_text: str
    contract_type: Optional[str] = None
    model_version: Optional[str] = None
    top_k: int = 3


class BatchPredictRequest(BaseModel):
    clauses: List[Dict[str, Any]]
    model_version: Optional[str] = None
    top_k: int = 3


class CorrectionRequest(BaseModel):
    clause_text: str
    predicted_label: str
    corrected_label: str
    reason: str
    model_version: Optional[str] = None
    batch_id: Optional[str] = None
    batch_item_id: Optional[int] = None
    operator: Optional[str] = None
    role: Optional[str] = "admin"


class RevertCorrectionRequest(BaseModel):
    correction_id: int
    batch_id: Optional[str] = None
    batch_item_id: Optional[int] = None
    operator: Optional[str] = None
    role: Optional[str] = "admin"


class TrainRequest(BaseModel):
    dataset_version: str
    hyperparams: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    auto_activate: bool = False


class EvaluateRequest(BaseModel):
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    test_ratio: float = 0.2
    random_state: int = 42


class FilterSchemeCreateRequest(BaseModel):
    name: str
    is_default: bool = False
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    created_from: Optional[str] = None
    created_to: Optional[str] = None
    source_type: Optional[str] = None
    has_conflicts: Optional[str] = None
    has_corrections: Optional[str] = None
    owner: str = "default"


class FilterSchemeUpdateRequest(BaseModel):
    name: Optional[str] = None
    is_default: Optional[bool] = None
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    created_from: Optional[str] = None
    created_to: Optional[str] = None
    source_type: Optional[str] = None
    has_conflicts: Optional[str] = None
    has_corrections: Optional[str] = None
    owner: str = "default"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "sample_data", "contracts_sample.csv")


@app.get("/sample_data/contracts_sample.csv")
async def sample_data_csv():
    if not os.path.exists(SAMPLE_DATA_PATH):
        raise HTTPException(status_code=404, detail="样例数据文件不存在")
    return FileResponse(
        SAMPLE_DATA_PATH,
        media_type="text/csv",
        filename="contracts_sample.csv",
    )


@app.get("/api/info")
async def api_info():
    active = get_active_model_version()
    datasets = list_dataset_versions()
    models = list_model_versions()
    return {
        "allowed_labels": ALLOWED_LABELS,
        "default_hyperparams": DEFAULT_HYPERPARAMS,
        "dataset_count": len(datasets),
        "model_count": len(models),
        "active_model": active,
        "dataset_latest": datasets[0] if datasets else None,
    }


@app.post("/api/datasets/validate")
async def api_validate_csv(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        result: ValidationResult = validate_csv(tmp.name)
        return {"filename": file.filename, **result.to_dict()}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post("/api/datasets/import")
async def api_import_csv(
    file: UploadFile = File(...),
    note: Optional[str] = Form(None),
    drop_empty_text: bool = Form(True),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        try:
            version, validation, extra = import_csv(
                filepath=tmp.name,
                filename=file.filename or "unknown.csv",
                note=note,
                drop_empty_text=drop_empty_text,
            )
            return {
                "success": True,
                "dataset_version": version,
                "filename": file.filename,
                "validation": validation.to_dict(),
                **extra,
            }
        except DataImportError as e:
            vr = validate_csv(tmp.name)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": str(e),
                    "validation": vr.to_dict(),
                },
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.get("/api/datasets")
async def api_list_datasets():
    return list_dataset_versions()


@app.get("/api/datasets/{version}")
async def api_get_dataset(version: str):
    ds = get_dataset_version(version)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"数据集版本不存在: {version}")
    return ds


@app.get("/api/datasets/{version}/download")
async def api_download_dataset(version: str):
    ds = get_dataset_version(version)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"数据集版本不存在: {version}")
    path = ds["stored_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="数据集文件已丢失")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{ds['filename'].rsplit('.', 1)[0]}_{version}.csv",
    )


@app.get("/api/models")
async def api_list_models():
    return list_model_versions()


@app.get("/api/models/active")
async def api_get_active_model():
    active = get_active_model_version()
    if active is None:
        return {"active": None}
    return {"active": active}


@app.get("/api/models/{version}")
async def api_get_model(version: str):
    mv = get_model_version(version)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"模型版本不存在: {version}")
    return mv


@app.post("/api/models/train")
async def api_train(req: TrainRequest):
    if not req.dataset_version:
        raise HTTPException(status_code=400, detail="dataset_version 不能为空")
    ds = get_dataset_version(req.dataset_version)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"数据集版本不存在: {req.dataset_version}")
    try:
        result = train_model(
            dataset_version=req.dataset_version,
            hyperparams=req.hyperparams,
            note=req.note,
            auto_activate=req.auto_activate,
        )
        return {"success": True, **result}
    except TrainingError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "dataset_version": req.dataset_version,
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/api/models/{version}/activate")
async def api_activate_model(
    version: str,
    operator_note: Optional[str] = Form(None),
):
    mv = get_model_version(version)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"模型版本不存在: {version}")
    if mv["training_status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"模型训练未完成（状态: {mv['training_status']}），不能激活",
        )
    ok = activate_model_version(version, operator_note=operator_note)
    if not ok:
        raise HTTPException(status_code=500, detail="激活失败")
    return {"success": True, "activated_version": version, "operator_note": operator_note}


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    try:
        result = predict_single(
            clause_text=req.clause_text,
            contract_type=req.contract_type,
            model_version=req.model_version,
            top_k=req.top_k,
        )
        return {"success": True, **result}
    except PredictionError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/predict/batch")
async def api_predict_batch(req: BatchPredictRequest):
    try:
        results = predict_batch(
            clauses=req.clauses,
            model_version=req.model_version,
            top_k=req.top_k,
        )
        return {"success": True, "total": len(results), "predictions": results}
    except PredictionError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/predict/csv")
async def api_predict_csv(
    file: UploadFile = File(...),
    model_version: Optional[str] = Form(None),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        try:
            result = predict_csv(dataset_path=tmp.name, model_version=model_version)
            return {"success": True, "filename": file.filename, **result}
        except PredictionError as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": str(e)},
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post("/api/corrections")
async def api_add_correction(req: CorrectionRequest):
    try:
        result = record_correction(
            clause_text=req.clause_text,
            predicted_label=req.predicted_label,
            corrected_label=req.corrected_label,
            reason=req.reason,
            model_version=req.model_version,
            batch_id=req.batch_id,
            batch_item_id=req.batch_item_id,
            operator=req.operator,
            role=req.role,
        )
        return {"success": True, **result}
    except PredictionError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/corrections/{correction_id}/revert")
async def api_revert_correction(correction_id: int, req: RevertCorrectionRequest):
    try:
        result = revert_correction(
            correction_id=correction_id,
            batch_id=req.batch_id,
            batch_item_id=req.batch_item_id,
            operator=req.operator,
            role=req.role,
        )
        return {"success": True, **result}
    except PredictionError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


@app.get("/api/corrections/history")
async def api_list_correction_history(
    correction_id: Optional[int] = None,
    batch_id: Optional[str] = None,
    limit: int = 100,
):
    return list_correction_history(
        correction_id=correction_id,
        batch_id=batch_id,
        limit=limit,
    )


@app.get("/api/corrections")
async def api_list_corrections(model_version: Optional[str] = None):
    return list_corrections(model_version=model_version)


@app.post("/api/evaluations")
async def api_run_evaluation(req: EvaluateRequest):
    try:
        result = run_evaluation(
            model_version=req.model_version,
            dataset_version=req.dataset_version,
            test_ratio=req.test_ratio,
            random_state=req.random_state,
        )
        return {"success": True, **result}
    except EvaluationError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            },
        )


@app.get("/api/evaluations")
async def api_list_evaluations(model_version: Optional[str] = None):
    return eval_list(model_version=model_version)


@app.get("/api/evaluations/{eval_id}")
async def api_get_evaluation(eval_id: int):
    detail = get_evaluation_detail(eval_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"评估记录不存在: {eval_id}")
    return detail


@app.get("/api/evaluations/{eval_id}/report.txt")
async def api_evaluation_report_txt(eval_id: int):
    detail = get_evaluation_detail(eval_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"评估记录不存在: {eval_id}")
    text = build_summary_text(detail)
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@app.get("/api/evaluations/{eval_id}/report.json")
async def api_evaluation_report_json(eval_id: int):
    detail = get_evaluation_detail(eval_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"评估记录不存在: {eval_id}")
    return JSONResponse(
        content={
            "success": True,
            "evaluation": detail,
            "plain_summary": build_summary_text(detail),
        }
    )


@app.get("/api/rollback-logs")
async def api_list_rollback_logs():
    return list_rollback_logs()


@app.post("/api/batches")
async def api_create_batch(
    file: UploadFile = File(...),
    model_version: Optional[str] = Form(None),
    note: Optional[str] = Form(None),
    source_type: str = Form("manual_upload"),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        try:
            result = create_batch_prediction(
                csv_path=tmp.name,
                filename=file.filename or "unknown.csv",
                model_version=model_version,
                note=note,
                top_k=3,
                source_type=source_type or "manual_upload",
            )
            return {"success": True, **result}
        except BatchError as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": str(e)},
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.get("/api/batches/source_types")
async def api_list_batch_source_types():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT source_type FROM batch_predictions WHERE source_type IS NOT NULL ORDER BY source_type"
        )
        return [r[0] for r in cur.fetchall() if r[0]]
    finally:
        conn.close()


@app.get("/api/batches")
async def api_list_batches(
    model_version: Optional[str] = None,
    dataset_version: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    source_type: Optional[str] = None,
    has_conflicts: Optional[str] = None,
    has_corrections: Optional[str] = None,
    limit: int = 50,
    role: str = "admin",
):
    try:
        return list_batches(
            model_version=model_version,
            dataset_version=dataset_version,
            created_from=created_from,
            created_to=created_to,
            source_type=source_type,
            has_conflicts=has_conflicts,
            has_corrections=has_corrections,
            limit=limit,
            role=role,
        )
    except BatchError as e:
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": str(e)},
        )


@app.get("/api/batches/{batch_id}")
async def api_get_batch(batch_id: str):
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"批次不存在: {batch_id}")
    counts = get_batch_item_count(batch_id)
    return {**batch, "item_counts": counts}


@app.get("/api/batches/{batch_id}/detail")
async def api_get_batch_detail(
    batch_id: str,
    role: str = "admin",
):
    try:
        detail = get_batch_detail(batch_id, role=role)
    except BatchError as e:
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": str(e)},
        )
    if detail is None:
        raise HTTPException(status_code=404, detail=f"批次不存在: {batch_id}")
    return detail


@app.get("/api/batches/{batch_id}/items")
async def api_get_batch_items(
    batch_id: str,
    include_conflicts: bool = True,
    limit: int = 200,
    offset: int = 0,
):
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"批次不存在: {batch_id}")
    items = get_batch_items(
        batch_id=batch_id,
        include_conflicts=include_conflicts,
        limit=limit,
        offset=offset,
    )
    counts = get_batch_item_count(batch_id)
    return {
        "batch_id": batch_id,
        "items": items,
        "total": counts["total"],
        "predicted": counts["predicted"],
        "conflicts": counts["conflicts"],
        "corrected": counts["corrected"],
    }


@app.post("/api/batches/{batch_id}/export/{export_type}")
async def api_export_batch(
    batch_id: str,
    export_type: str,
    note: Optional[str] = Form(None),
    role: str = Form("admin"),
):
    if export_type not in ("prediction", "training", "audit"):
        raise HTTPException(status_code=400, detail=f"不支持的导出类型: {export_type}")
    try:
        if export_type == "audit":
            result = export_audit_to_csv(
                batch_id=batch_id,
                note=note,
                role=role,
            )
        else:
            result = export_batch_to_csv(
                batch_id=batch_id,
                export_type=export_type,
                note=note,
            )
        return {"success": True, **result}
    except BatchError as e:
        return JSONResponse(
            status_code=400 if export_type != "audit" else 403,
            content={"success": False, "error": str(e)},
        )


@app.get("/api/batches/{batch_id}/export/{export_type}/download")
async def api_download_batch_export(
    batch_id: str,
    export_type: str,
    role: str = "admin",
):
    if export_type not in ("prediction", "training", "audit"):
        raise HTTPException(status_code=400, detail=f"不支持的导出类型: {export_type}")
    try:
        if export_type == "audit":
            result = export_audit_to_csv(
                batch_id=batch_id,
                note=None,
                role=role,
            )
        else:
            result = export_batch_to_csv(
                batch_id=batch_id,
                export_type=export_type,
                note=None,
            )
    except BatchError as e:
        raise HTTPException(status_code=400 if export_type != "audit" else 403, detail=str(e))

    file_path = result["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="导出文件不存在")

    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=result["filename"],
    )


# ========== Filter Schemes APIs ==========

@app.get("/api/filter-schemes")
async def api_list_filter_schemes(owner: str = "default", role: str = "admin"):
    if not check_permission(role, "batch_view"):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": f"角色 '{role}' 没有查看权限"},
        )
    return list_filter_schemes(owner=owner)


@app.get("/api/filter-schemes/default")
async def api_get_default_filter_scheme(owner: str = "default", role: str = "admin"):
    if not check_permission(role, "batch_view"):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": f"角色 '{role}' 没有查看权限"},
        )
    scheme = get_default_filter_scheme(owner=owner)
    if scheme is None:
        return {"default": None}
    return {"default": scheme}


@app.get("/api/filter-schemes/{scheme_id}")
async def api_get_filter_scheme(scheme_id: int, owner: str = "default", role: str = "admin"):
    if not check_permission(role, "batch_view"):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": f"角色 '{role}' 没有查看权限"},
        )
    scheme = get_filter_scheme(scheme_id, owner=owner)
    if scheme is None:
        raise HTTPException(status_code=404, detail=f"筛选方案不存在: {scheme_id}")
    return scheme


@app.post("/api/filter-schemes")
async def api_create_filter_scheme(req: FilterSchemeCreateRequest, role: str = "admin"):
    if not check_permission(role, "filter_scheme_manage"):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": f"角色 '{role}' 没有管理筛选方案的权限"},
        )
    if not req.name or not req.name.strip():
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "筛选方案名称不能为空"},
        )
    try:
        scheme = create_filter_scheme(
            name=req.name.strip(),
            owner=req.owner,
            is_default=req.is_default,
            model_version=req.model_version,
            dataset_version=req.dataset_version,
            created_from=req.created_from,
            created_to=req.created_to,
            source_type=req.source_type,
            has_conflicts=req.has_conflicts,
            has_corrections=req.has_corrections,
        )
        return {"success": True, **scheme}
    except ValueError as e:
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": str(e)},
        )


@app.put("/api/filter-schemes/{scheme_id}")
async def api_update_filter_scheme(scheme_id: int, req: FilterSchemeUpdateRequest, role: str = "admin"):
    if not check_permission(role, "filter_scheme_manage"):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": f"角色 '{role}' 没有管理筛选方案的权限"},
        )
    try:
        scheme = update_filter_scheme(
            scheme_id=scheme_id,
            owner=req.owner,
            name=req.name.strip() if req.name else None,
            is_default=req.is_default,
            model_version=req.model_version,
            dataset_version=req.dataset_version,
            created_from=req.created_from,
            created_to=req.created_to,
            source_type=req.source_type,
            has_conflicts=req.has_conflicts,
            has_corrections=req.has_corrections,
        )
        return {"success": True, **scheme}
    except ValueError as e:
        return JSONResponse(
            status_code=404 if "不存在" in str(e) else 409,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/filter-schemes/{scheme_id}/default")
async def api_set_default_filter_scheme(scheme_id: int, owner: str = "default", role: str = "admin"):
    if not check_permission(role, "filter_scheme_manage"):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": f"角色 '{role}' 没有管理筛选方案的权限"},
        )
    ok = set_default_filter_scheme(scheme_id, owner=owner)
    if not ok:
        raise HTTPException(status_code=404, detail=f"筛选方案不存在: {scheme_id}")
    return {"success": True, "scheme_id": scheme_id, "is_default": True}


@app.delete("/api/filter-schemes/{scheme_id}")
async def api_delete_filter_scheme(scheme_id: int, owner: str = "default", role: str = "admin"):
    if not check_permission(role, "filter_scheme_manage"):
        return JSONResponse(
            status_code=403,
            content={"success": False, "error": f"角色 '{role}' 没有管理筛选方案的权限"},
        )
    ok = delete_filter_scheme(scheme_id, owner=owner)
    if not ok:
        raise HTTPException(status_code=404, detail=f"筛选方案不存在: {scheme_id}")
    return {"success": True, "scheme_id": scheme_id}


@app.get("/api/exports")
async def api_list_exports(
    batch_id: Optional[str] = None,
    export_type: Optional[str] = None,
    limit: int = 50,
):
    return list_exports(batch_id=batch_id, export_type=export_type, limit=limit)


@app.get("/api/exports/{export_id}/download")
async def api_download_export(export_id: str):
    exp = get_export(export_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"导出记录不存在: {export_id}")
    file_path = exp["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="导出文件已丢失")
    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=exp["filename"],
    )


@app.get("/api/operation-logs")
async def api_list_operation_logs(
    operation_type: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = 100,
):
    return list_operation_logs(
        operation_type=operation_type,
        entity_type=entity_type,
        entity_id=entity_id,
        limit=limit,
    )


@app.get("/api/permissions")
async def api_list_permissions():
    return list_permissions()


@app.get("/api/permissions/check")
async def api_check_permission(role: str, operation: str):
    allowed = check_permission(role, operation)
    return {"role": role, "operation": operation, "allowed": allowed}


@app.get("/health")
async def health_check():
    active = get_active_model_version()
    try:
        batch_count = len(list_batches(limit=1000, role=None))
    except Exception:
        batch_count = 0
    try:
        export_count = len(list_exports(limit=1000))
    except Exception:
        export_count = 0
    return {
        "status": "ok",
        "db_exists": os.path.exists(os.path.join(BASE_DIR, "data", "app.db")),
        "active_model": active["version"] if active else None,
        "datasets": len(list_dataset_versions()),
        "models": len(list_model_versions()),
        "evaluations": len(list_evaluations()),
        "corrections": len(list_corrections()),
        "batches": batch_count,
        "exports": export_count,
    }

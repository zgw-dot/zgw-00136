import os
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from .version_manager import (
    get_model_version,
    get_active_model_version,
    add_correction,
)
from .trainer import _build_features_from_dataframe


class PredictionError(Exception):
    pass


class ModelBundle:
    def __init__(self, classifier, vectorizer, label_encoder, model_info: Dict[str, Any]):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.model_info = model_info

    @property
    def version(self) -> str:
        return self.model_info["version"]

    @property
    def dataset_version(self) -> str:
        return self.model_info["dataset_version"]

    @property
    def classes(self) -> List[str]:
        return self.label_encoder.classes_.tolist()


def load_model_bundle(model_version: Optional[str] = None) -> ModelBundle:
    if model_version:
        info = get_model_version(model_version)
    else:
        info = get_active_model_version()

    if info is None:
        if model_version:
            raise PredictionError(f"模型版本不存在: {model_version}")
        else:
            raise PredictionError("尚无激活的模型，请先训练并激活一个模型版本")

    if info["training_status"] != "completed":
        raise PredictionError(
            f"模型版本 {model_version} 训练状态为 '{info['training_status']}'，无法用于预测"
        )

    required_paths = ["model_path", "vectorizer_path", "label_encoder_path"]
    for p in required_paths:
        if not os.path.exists(info[p]):
            raise PredictionError(
                f"模型文件缺失: {info[p]}，模型版本可能已损坏"
            )

    try:
        classifier = joblib.load(info["model_path"])
        vectorizer = joblib.load(info["vectorizer_path"])
        label_encoder = joblib.load(info["label_encoder_path"])
    except Exception as e:
        raise PredictionError(f"加载模型文件失败: {e}") from e

    return ModelBundle(classifier, vectorizer, label_encoder, info)


def _build_text_features(clause_text: str, contract_type: Optional[str] = None) -> str:
    if contract_type and str(contract_type).strip():
        return f"[{str(contract_type).strip()}] {clause_text}"
    return clause_text


def predict_single(
    clause_text: str,
    contract_type: Optional[str] = None,
    model_version: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    if not clause_text or not str(clause_text).strip():
        raise PredictionError("条款文本不能为空")

    bundle = load_model_bundle(model_version)

    text = _build_text_features(clause_text, contract_type)
    X = bundle.vectorizer.transform([text])

    probs = bundle.classifier.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = bundle.label_encoder.inverse_transform([pred_idx])[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = []
    for i in top_indices:
        top_results.append({
            "label": bundle.label_encoder.inverse_transform([int(i)])[0],
            "probability": float(probs[i]),
        })

    return {
        "clause_text": clause_text,
        "contract_type": contract_type,
        "predicted_label": pred_label,
        "confidence": float(probs[pred_idx]),
        "top_predictions": top_results,
        "model_version": bundle.version,
        "dataset_version": bundle.dataset_version,
        "all_classes": bundle.classes,
    }


def predict_batch(
    clauses: List[Dict[str, Any]],
    model_version: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if not clauses:
        return []

    bundle = load_model_bundle(model_version)

    texts = []
    for c in clauses:
        t = c.get("条款文本") or c.get("text") or ""
        ct = c.get("合同类型") or c.get("contract_type")
        texts.append(_build_text_features(t, ct))

    X = bundle.vectorizer.transform(texts)
    probs_all = bundle.classifier.predict_proba(X)
    pred_indices = np.argmax(probs_all, axis=1)
    pred_labels = bundle.label_encoder.inverse_transform(pred_indices)

    results = []
    for i, c in enumerate(clauses):
        probs = probs_all[i]
        pred_idx = int(pred_indices[i])
        top_inds = np.argsort(probs)[::-1][:top_k]
        top_results = []
        for j in top_inds:
            top_results.append({
                "label": bundle.label_encoder.inverse_transform([int(j)])[0],
                "probability": float(probs[j]),
            })

        results.append({
            "clause_text": c.get("条款文本") or c.get("text") or "",
            "contract_type": c.get("合同类型") or c.get("contract_type"),
            "predicted_label": pred_labels[i],
            "confidence": float(probs[pred_idx]),
            "top_predictions": top_results,
            "model_version": bundle.version,
            "dataset_version": bundle.dataset_version,
        })

    return results


def predict_csv(
    dataset_path: str,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    bundle = load_model_bundle(model_version)

    try:
        try:
            df = pd.read_csv(dataset_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(dataset_path, encoding="gbk")
    except Exception as e:
        raise PredictionError(f"读取CSV失败: {e}") from e

    if "条款文本" not in df.columns:
        raise PredictionError("CSV缺少必填列: 条款文本")

    df = df[df["条款文本"].astype(str).str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)

    if len(df) == 0:
        return {"total": 0, "predictions": [], "model_version": bundle.version}

    texts = _build_features_from_dataframe(df, use_extra=True)
    X = bundle.vectorizer.transform(texts)
    probs_all = bundle.classifier.predict_proba(X)
    pred_indices = np.argmax(probs_all, axis=1)
    pred_labels = bundle.label_encoder.inverse_transform(pred_indices)

    predictions = []
    for i, row in df.iterrows():
        probs = probs_all[i]
        pred_idx = int(pred_indices[i])
        predictions.append({
            "row_index": i,
            "clause_text": str(row["条款文本"]),
            "contract_type": str(row.get("合同类型", "")) if "合同类型" in df.columns else "",
            "date": str(row.get("时间", "")) if "时间" in df.columns else "",
            "true_label": str(row.get("风险标签", "")) if "风险标签" in df.columns else "",
            "predicted_label": pred_labels[i],
            "confidence": float(probs[pred_idx]),
        })

    return {
        "total": len(predictions),
        "predictions": predictions,
        "model_version": bundle.version,
        "dataset_version": bundle.dataset_version,
    }


def record_correction(
    clause_text: str,
    predicted_label: str,
    corrected_label: str,
    reason: str,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    if not clause_text or not str(clause_text).strip():
        raise PredictionError("条款文本不能为空")
    if not corrected_label or not str(corrected_label).strip():
        raise PredictionError("人工改判标签不能为空")
    if not reason or not str(reason).strip():
        raise PredictionError("必须填写改判原因")

    if model_version is None:
        active = get_active_model_version()
        if active is None:
            raise PredictionError("没有激活的模型，无法记录改判归属")
        model_version = active["version"]

    mv = get_model_version(model_version)
    if mv is None:
        raise PredictionError(f"模型版本不存在: {model_version}")

    correction_id = add_correction(
        model_version=model_version,
        clause_text=str(clause_text).strip(),
        predicted_label=str(predicted_label).strip(),
        corrected_label=str(corrected_label).strip(),
        reason=str(reason).strip(),
    )

    return {
        "correction_id": correction_id,
        "model_version": model_version,
        "clause_text": clause_text,
        "predicted_label": predicted_label,
        "corrected_label": corrected_label,
        "reason": reason,
    }

import os
import traceback
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .data_manager import load_labeled_data
from .version_manager import (
    get_model_version,
    get_dataset_version,
    get_active_model_version,
    save_evaluation,
    list_evaluations as _list_evaluations,
    get_evaluation as _get_evaluation,
)
from .predictor import load_model_bundle, PredictionError
from .trainer import _build_features_from_dataframe


class EvaluationError(Exception):
    pass


def run_evaluation(
    model_version: Optional[str] = None,
    dataset_version: Optional[str] = None,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    if model_version:
        mv = get_model_version(model_version)
    else:
        mv = get_active_model_version()

    if mv is None:
        raise EvaluationError(
            f"模型版本不存在: {model_version}" if model_version else "尚无激活的模型"
        )
    if mv["training_status"] != "completed":
        raise EvaluationError(
            f"模型 {mv['version']} 训练状态为 {mv['training_status']}，无法评估"
        )
    model_version = mv["version"]

    if dataset_version is None:
        dataset_version = mv["dataset_version"]

    ds = get_dataset_version(dataset_version)
    if ds is None:
        raise EvaluationError(f"数据集版本不存在: {dataset_version}")

    try:
        bundle = load_model_bundle(model_version)
    except PredictionError as e:
        raise EvaluationError(str(e)) from e

    try:
        df = load_labeled_data(ds["stored_path"])
        if len(df) == 0:
            raise EvaluationError("数据集中没有可用的带标签数据")

        texts = _build_features_from_dataframe(df, use_extra=True)
        y_true_raw = df["风险标签"].tolist()

        try:
            y_true = bundle.label_encoder.transform(y_true_raw)
        except ValueError as ve:
            unseen = set(y_true_raw) - set(bundle.label_encoder.classes_)
            raise EvaluationError(
                f"数据集包含模型未见的标签: {unseen}，无法使用该模型评估"
            ) from ve

        X = bundle.vectorizer.transform(texts)
        y_pred = bundle.classifier.predict(X)
        y_proba = bundle.classifier.predict_proba(X)

        total = len(y_true)
        n_test = max(1, int(total * test_ratio))
        if n_test < 5:
            n_test = total

        rng = np.random.RandomState(random_state)
        indices = np.arange(total)
        rng.shuffle(indices)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        y_test_true = np.array(y_true)[test_idx]
        y_test_pred = y_pred[test_idx]
        test_texts = [texts[i] for i in test_idx]

        labels = list(bundle.label_encoder.classes_)
        label_indices = list(range(len(labels)))

        accuracy = float(accuracy_score(y_test_true, y_test_pred))
        precision_macro = float(
            precision_score(y_test_true, y_test_pred, average="macro", zero_division=0)
        )
        recall_macro = float(
            recall_score(y_test_true, y_test_pred, average="macro", zero_division=0)
        )
        f1_macro = float(
            f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
        )

        report_dict = classification_report(
            y_test_true,
            y_test_pred,
            labels=label_indices,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test_true, y_test_pred, labels=label_indices).tolist()

        per_class_samples = {}
        for label, idx in zip(labels, label_indices):
            mask = np.array(y_true) == idx
            per_class_samples[label] = int(mask.sum())

        eval_id, report_path = save_evaluation(
            model_version=model_version,
            dataset_version=dataset_version,
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            report_json=report_dict,
            confusion_matrix=cm,
            labels=labels,
            train_samples=len(train_idx),
            test_samples=len(test_idx),
        )

        misclassified = []
        for j, (ti, true_idx, pred_idx) in enumerate(
            zip(test_idx, y_test_true, y_test_pred)
        ):
            if true_idx != pred_idx:
                misclassified.append({
                    "test_position": j,
                    "original_index": int(ti),
                    "clause_text": df.iloc[int(ti)]["条款文本"],
                    "contract_type": (
                        df.iloc[int(ti)].get("合同类型", "")
                        if "合同类型" in df.columns else ""
                    ),
                    "true_label": labels[int(true_idx)],
                    "predicted_label": labels[int(pred_idx)],
                    "confidence_true": float(y_proba[int(ti)][int(true_idx)]),
                    "confidence_pred": float(y_proba[int(ti)][int(pred_idx)]),
                })

        return {
            "evaluation_id": eval_id,
            "report_path": report_path,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "dataset_filename": ds["filename"],
            "model_created_at": mv.get("created_at"),
            "dataset_created_at": ds["created_at"],
            "metrics": {
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
            },
            "labels": labels,
            "per_class_samples": per_class_samples,
            "label_distribution_train_test": {
                "train": {
                    labels[i]: int((np.array(y_true)[train_idx] == i).sum())
                    for i in label_indices
                },
                "test": {
                    labels[i]: int((np.array(y_true)[test_idx] == i).sum())
                    for i in label_indices
                },
            },
            "classification_report": report_dict,
            "confusion_matrix": cm,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "total_samples": total,
            "misclassified_count": len(misclassified),
            "misclassified_examples": misclassified[:50],
            "traceability": {
                "model_version": model_version,
                "dataset_version": dataset_version,
                "dataset_file": ds["filename"],
                "dataset_version_note": ds.get("note"),
                "model_hyperparams": mv.get("hyperparams"),
                "model_feature_count": mv.get("feature_count"),
            },
        }

    except EvaluationError:
        raise
    except Exception as e:
        raise EvaluationError(
            f"评估异常: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        ) from e


def list_evaluations(model_version: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = _list_evaluations(model_version)
    for r in rows:
        r["traceability"] = {
            "evaluation_id": r["id"],
            "model_version": r["model_version"],
            "dataset_version": r["dataset_version"],
        }
    return rows


def get_evaluation_detail(eval_id: int) -> Optional[Dict[str, Any]]:
    r = _get_evaluation(eval_id)
    if r is None:
        return None

    mv = get_model_version(r["model_version"])
    ds = get_dataset_version(r["dataset_version"])

    result = dict(r)
    result["metrics_summary"] = {
        "accuracy": r["accuracy"],
        "precision_macro": r["precision_macro"],
        "recall_macro": r["recall_macro"],
        "f1_macro": r["f1_macro"],
    }
    result["traceability"] = {
        "evaluation_id": eval_id,
        "model_version": r["model_version"],
        "dataset_version": r["dataset_version"],
        "model_training_status": mv["training_status"] if mv else None,
        "model_hyperparams": mv["hyperparams"] if mv else None,
        "model_feature_count": mv["feature_count"] if mv else None,
        "model_created_at": mv["created_at"] if mv else None,
        "dataset_filename": ds["filename"] if ds else None,
        "dataset_row_count": ds["row_count"] if ds else None,
        "dataset_labeled_count": ds["labeled_count"] if ds else None,
        "dataset_created_at": ds["created_at"] if ds else None,
        "dataset_label_distribution": ds["label_distribution"] if ds else None,
        "evaluation_created_at": r["created_at"],
    }
    result["per_class_metrics"] = {
        label: r["report_json"].get(label, {})
        for label in r["labels"]
    }
    return result


def build_summary_text(eval_detail: Dict[str, Any]) -> str:
    t = eval_detail["traceability"]
    m = eval_detail["metrics_summary"]
    lines = []
    lines.append("=" * 60)
    lines.append("合同条款风险分类器评估报告")
    lines.append("=" * 60)
    lines.append(f"评估ID:        {t['evaluation_id']}")
    lines.append(f"评估时间:      {t['evaluation_created_at']}")
    lines.append("")
    lines.append("【版本追溯】")
    lines.append(f"  模型版本:    {t['model_version']}")
    lines.append(f"  训练时间:    {t['model_created_at']}")
    lines.append(f"  特征数量:    {t['model_feature_count']}")
    lines.append(f"  数据集版本:  {t['dataset_version']}")
    lines.append(f"  数据文件:    {t['dataset_filename']}")
    lines.append(f"  数据规模:    {t['dataset_row_count']}行（含标签{t['dataset_labeled_count']}条）")
    lines.append(f"  数据导入:    {t['dataset_created_at']}")
    lines.append("")
    lines.append("【整体指标】")
    lines.append(f"  Accuracy:    {m['accuracy']:.4f}")
    lines.append(f"  Precision:   {m['precision_macro']:.4f} (macro)")
    lines.append(f"  Recall:      {m['recall_macro']:.4f} (macro)")
    lines.append(f"  F1:          {m['f1_macro']:.4f} (macro)")
    lines.append("")
    lines.append("【各类别详情】")
    lines.append(f"  {'类别':<10}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}")
    for label, metrics in eval_detail["per_class_metrics"].items():
        if isinstance(metrics, dict) and "precision" in metrics:
            lines.append(
                f"  {label:<10}{metrics['precision']:>12.4f}"
                f"{metrics['recall']:>12.4f}{metrics['f1-score']:>12.4f}"
                f"{metrics['support']:>10}"
            )
    lines.append("")
    lines.append("【混淆矩阵】（行=真实，列=预测）")
    labels = eval_detail["labels"]
    header = " " * 12 + "".join(f"{l:>10}" for l in labels)
    lines.append(header)
    for i, row in enumerate(eval_detail["confusion_matrix"]):
        lines.append(f"  {labels[i]:<10}" + "".join(f"{v:>10}" for v in row))
    lines.append("")
    lines.append("【数据集标签分布】")
    dist = t.get("dataset_label_distribution") or {}
    for k, v in dist.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("【模型超参数】")
    hp = t.get("model_hyperparams") or {}
    lines.append(f"  {hp}")
    lines.append("=" * 60)
    return "\n".join(lines)

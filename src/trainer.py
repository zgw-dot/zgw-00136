import re
import traceback
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .data_manager import load_labeled_data
from .version_manager import (
    create_model_version_entry,
    update_model_version_status,
    get_dataset_version,
)


def _chinese_tokenizer(text: str) -> List[str]:
    text = str(text).lower()
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]+|\d+", text)
    if not tokens:
        tokens = list(text)
    return tokens


DEFAULT_HYPERPARAMS = {
    "vectorizer": {
        "max_features": 5000,
        "ngram_range": [1, 2],
        "min_df": 1,
    },
    "classifier": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "split": {
        "test_size": 0.2,
        "random_state": 42,
    },
}


class TrainingError(Exception):
    pass


def _build_features_from_dataframe(df: pd.DataFrame, use_extra: bool = True) -> List[str]:
    texts = df["条款文本"].astype(str).tolist()

    if not use_extra:
        return texts

    enhanced = []
    contract_types = df["合同类型"].astype(str).tolist() if "合同类型" in df.columns else [""] * len(df)
    for text, ct in zip(texts, contract_types):
        prefix = f"[{ct}] " if ct and ct.strip() else ""
        enhanced.append(prefix + text)
    return enhanced


def prepare_training_data(
    dataset_version: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    ds = get_dataset_version(dataset_version)
    if ds is None:
        raise TrainingError(f"数据集版本不存在: {dataset_version}")

    df = load_labeled_data(ds["stored_path"])
    if len(df) == 0:
        raise TrainingError("没有可用的带标签数据")

    labels = df["风险标签"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(labels)

    return df, y, le


def train_model(
    dataset_version: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    note: Optional[str] = None,
    auto_activate: bool = False,
) -> Dict[str, Any]:
    ds = get_dataset_version(dataset_version)
    if ds is None:
        raise TrainingError(f"数据集版本不存在: {dataset_version}")

    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS

    model_version, paths = create_model_version_entry(
        dataset_version=dataset_version,
        hyperparams=hyperparams,
        note=note,
    )

    try:
        df = load_labeled_data(ds["stored_path"])
        if len(df) < 5:
            raise TrainingError("标签样本不足（至少5条），无法进行有效训练")

        texts = _build_features_from_dataframe(df, use_extra=True)
        labels_raw = df["风险标签"].tolist()

        le = LabelEncoder()
        y = le.fit_transform(labels_raw)
        num_classes = len(le.classes_)
        if num_classes < 2:
            raise TrainingError(f"只有 {num_classes} 个类别，至少需要2个类别才能训练")

        split_cfg = hyperparams.get("split", DEFAULT_HYPERPARAMS["split"])
        test_size = split_cfg.get("test_size", 0.2)
        random_state = split_cfg.get("random_state", 42)

        stratify = y if num_classes >= 2 and min(pd.Series(y).value_counts()) >= 2 else None

        try:
            X_texts_train, X_texts_test, y_train, y_test, idx_train, idx_test = train_test_split(
                texts,
                y.tolist(),
                range(len(texts)),
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            X_texts_train, X_texts_test, y_train, y_test, idx_train, idx_test = train_test_split(
                texts,
                y.tolist(),
                range(len(texts)),
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

        vec_cfg = hyperparams.get("vectorizer", DEFAULT_HYPERPARAMS["vectorizer"])
        vectorizer = TfidfVectorizer(
            tokenizer=_chinese_tokenizer,
            token_pattern=None,
            max_features=vec_cfg.get("max_features", 5000),
            ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
            min_df=vec_cfg.get("min_df", 1),
            sublinear_tf=True,
        )

        X_train = vectorizer.fit_transform(X_texts_train)
        X_test = vectorizer.transform(X_texts_test)

        feature_count = X_train.shape[1]
        if feature_count == 0:
            raise TrainingError("TF-IDF未提取到任何有效特征，请检查条款文本内容")

        clf_cfg = hyperparams.get("classifier", DEFAULT_HYPERPARAMS["classifier"])
        use_class_weight = clf_cfg.get("class_weight", "balanced")
        classes_arr = np.unique(y_train)
        try:
            cw_arr = compute_class_weight("balanced", classes=classes_arr, y=y_train)
            class_weight_dict = {c: w for c, w in zip(classes_arr, cw_arr)}
        except Exception:
            class_weight_dict = None

        solver = clf_cfg.get("solver", "lbfgs")
        if num_classes >= 3 and solver == "liblinear":
            solver = "lbfgs"

        classifier = LogisticRegression(
            C=clf_cfg.get("C", 1.0),
            class_weight=class_weight_dict if use_class_weight else None,
            max_iter=clf_cfg.get("max_iter", 1000),
            solver=solver,
            random_state=random_state,
        )

        classifier.fit(X_train, y_train)

        train_pred = classifier.predict(X_train)
        test_pred = classifier.predict(X_test)
        train_acc = float(np.mean(train_pred == np.array(y_train)))
        test_acc = float(np.mean(test_pred == np.array(y_test)))

        if test_acc < 0.3:
            raise TrainingError(f"测试集准确率过低（{test_acc:.3f}），模型训练可能未收敛，请检查数据质量")

        joblib.dump(classifier, paths["model_path"])
        joblib.dump(vectorizer, paths["vectorizer_path"])
        joblib.dump(le, paths["label_encoder_path"])

        update_model_version_status(
            version=model_version,
            status="completed",
            error_message=None,
            feature_count=feature_count,
        )

        result = {
            "model_version": model_version,
            "dataset_version": dataset_version,
            "feature_count": feature_count,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "num_classes": num_classes,
            "classes": le.classes_.tolist(),
            "train_samples": len(idx_train),
            "test_samples": len(idx_test),
            "label_counts": df["风险标签"].value_counts().to_dict(),
            "auto_activated": False,
            "paths": paths,
            "split_indices": {"train": list(idx_train), "test": list(idx_test)},
            "label_encoder_classes": le.classes_.tolist(),
            "y_test": list(y_test),
            "test_predictions": test_pred.tolist(),
            "X_test_texts": X_texts_test,
        }

        if auto_activate:
            from .version_manager import activate_model_version
            ok = activate_model_version(model_version, operator_note="训练完成自动激活")
            result["auto_activated"] = ok

        return result

    except TrainingError:
        update_model_version_status(
            version=model_version,
            status="failed",
            error_message=str(traceback.format_exc()),
            feature_count=None,
        )
        raise
    except Exception as e:
        update_model_version_status(
            version=model_version,
            status="failed",
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            feature_count=None,
        )
        raise TrainingError(f"训练异常: {e}") from e

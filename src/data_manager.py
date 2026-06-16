import os
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from .version_manager import save_dataset_version

REQUIRED_COLUMNS = ["条款文本", "风险标签"]
OPTIONAL_COLUMNS = ["合同类型", "时间"]
ALLOWED_LABELS = ["付款", "违约", "自动续约"]


class DataImportError(Exception):
    pass


class ValidationResult:
    def __init__(self):
        self.valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.row_count: int = 0
        self.valid_rows: int = 0
        self.labeled_count: int = 0
        self.empty_text_count: int = 0
        self.empty_label_count: int = 0
        self.invalid_label_count: int = 0
        self.duplicate_count: int = 0
        self.label_distribution: Dict[str, int] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_count": self.row_count,
            "valid_rows": self.valid_rows,
            "labeled_count": self.labeled_count,
            "empty_text_count": self.empty_text_count,
            "empty_label_count": self.empty_label_count,
            "invalid_label_count": self.invalid_label_count,
            "duplicate_count": self.duplicate_count,
            "label_distribution": self.label_distribution,
        }


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def validate_csv(filepath: str) -> ValidationResult:
    result = ValidationResult()

    if not os.path.exists(filepath):
        result.valid = False
        result.errors.append(f"文件不存在: {filepath}")
        return result

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding="gbk")
        except Exception as e:
            result.valid = False
            result.errors.append(f"无法解析CSV文件，请检查编码(UTF-8/GBK): {e}")
            return result
    except Exception as e:
        result.valid = False
        result.errors.append(f"读取CSV失败: {e}")
        return result

    result.row_count = len(df)

    columns = list(df.columns)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_required:
        result.valid = False
        result.errors.append(
            f"缺少必填列: {', '.join(missing_required)}。必填列为: {', '.join(REQUIRED_COLUMNS)}"
        )

    for col in OPTIONAL_COLUMNS:
        if col not in columns:
            result.warnings.append(f"缺少可选列: {col}（建议包含以便增强特征）")

    if not result.valid:
        return result

    if result.row_count == 0:
        result.valid = False
        result.errors.append("CSV文件为空，没有数据行")
        return result

    valid_mask = pd.Series([True] * len(df), index=df.index)

    empty_text_mask = ~df["条款文本"].apply(_is_nonempty)
    result.empty_text_count = int(empty_text_mask.sum())
    if result.empty_text_count > 0:
        result.warnings.append(
            f"存在 {result.empty_text_count} 行条款文本为空，将被排除"
        )
        valid_mask &= ~empty_text_mask

    label_mask = df["风险标签"].apply(_is_nonempty)
    result.empty_label_count = int((~label_mask).sum())
    if result.empty_label_count > 0:
        result.warnings.append(
            f"存在 {result.empty_label_count} 行风险标签为空，训练时将被排除"
        )

    labeled_df = df[valid_mask & label_mask].copy()

    if len(labeled_df) > 0:
        invalid_label_mask = ~labeled_df["风险标签"].isin(ALLOWED_LABELS)
        result.invalid_label_count = int(invalid_label_mask.sum())
        if result.invalid_label_count > 0:
            invalid_labels = sorted(
                labeled_df.loc[invalid_label_mask, "风险标签"].unique().tolist()
            )
            result.warnings.append(
                f"存在 {result.invalid_label_count} 行标签不在允许列表 {ALLOWED_LABELS} 中: {invalid_labels}"
            )
        labeled_df = labeled_df[~invalid_label_mask]

    result.labeled_count = len(labeled_df)

    if result.labeled_count == 0:
        result.valid = False
        result.errors.append("没有可用的带标签样本，无法训练")
        return result

    result.label_distribution = (
        labeled_df["风险标签"].value_counts().sort_index().to_dict()
    )

    min_label_count = min(result.label_distribution.values())
    if min_label_count < 3:
        result.warnings.append(
            f"部分类别样本过少（最少 {min_label_count} 条），可能影响分类效果"
        )

    text_series = df.loc[valid_mask, "条款文本"].astype(str).str.strip()
    dup_mask = text_series.duplicated(keep="first")
    result.duplicate_count = int(dup_mask.sum())
    if result.duplicate_count > 0:
        result.warnings.append(f"存在 {result.duplicate_count} 行重复条款文本")

    result.valid_rows = int(valid_mask.sum())
    return result


def import_csv(
    filepath: str,
    filename: str,
    note: Optional[str] = None,
    drop_empty_text: bool = True,
) -> Tuple[str, ValidationResult, Dict[str, Any]]:
    validation = validate_csv(filepath)
    if not validation.valid:
        raise DataImportError("; ".join(validation.errors))

    try:
        try:
            df = pd.read_csv(filepath, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="gbk")
    except Exception as e:
        raise DataImportError(f"读取CSV失败: {e}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    if drop_empty_text:
        df = df[df["条款文本"].apply(_is_nonempty)].copy()
        df.reset_index(drop=True, inplace=True)

    cleaned_path = filepath + ".cleaned.tmp.csv"
    df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    columns_info = list(df.columns)

    version = save_dataset_version(
        filename=filename,
        source_path=cleaned_path,
        row_count=len(df),
        labeled_count=validation.labeled_count,
        label_distribution=validation.label_distribution,
        columns_info=columns_info,
        note=note,
    )

    try:
        os.remove(cleaned_path)
    except OSError:
        pass

    return version, validation, {
        "dataset_version": version,
        "stored_filename": f"{version}.csv",
        "columns": columns_info,
    }


def load_labeled_data(dataset_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="gbk")

    df["条款文本"] = df["条款文本"].astype(str)
    df = df[df["条款文本"].apply(lambda x: x.strip() != "")]

    if "风险标签" in df.columns:
        df["风险标签"] = df["风险标签"].astype(str)
        df_labeled = df[
            df["风险标签"].apply(
                lambda x: x.strip() != "" and x.strip() in ALLOWED_LABELS
            )
        ].copy()
        df_labeled["风险标签"] = df_labeled["风险标签"].str.strip()
    else:
        df_labeled = df.iloc[0:0].copy()

    return df_labeled.reset_index(drop=True)


def load_all_data(dataset_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="gbk")

    df["条款文本"] = df["条款文本"].astype(str)
    df = df[df["条款文本"].apply(lambda x: x.strip() != "")]
    return df.reset_index(drop=True)

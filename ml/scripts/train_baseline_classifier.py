from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.utils.tracking import start_run, log_params, log_metrics, log_artifact  # noqa: E402


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_confusion_matrix(y_true, y_pred, labels, out_csv: Path, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(out_csv, encoding="utf-8")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(out_png.stem)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def ensure_text_column(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Приводит dataframe к ожидаемой текстовой колонке.
    Если text_column отсутствует, но есть 'text', создаёт text_column из 'text'.
    """
    if text_column in df.columns:
        return df

    if "text" in df.columns:
        df = df.copy()
        df[text_column] = df["text"].astype(str)
        return df

    raise KeyError(
        f"Expected text column '{text_column}', but dataframe has columns: {df.columns.tolist()}"
    )

def build_dataset(df: pd.DataFrame, text_column: str, label_column: str, label2id: dict):
    ds = Dataset.from_pandas(df[[text_column, label_column]].copy(), preserve_index=False)
    ds = ds.map(lambda x: {"labels": label2id[x[label_column]]})
    return ds


def tokenize_labeled_dataset(ds: Dataset, tokenizer, text_column: str, max_length: int):
    remove_cols = [c for c in ds.column_names if c not in ["labels"]]
    return ds.map(
        lambda batch: tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        batched=True,
        remove_columns=remove_cols,
    )


def tokenize_unlabeled_dataset(ds: Dataset, tokenizer, text_column: str, max_length: int):
    remove_cols = list(ds.column_names)
    return ds.map(
        lambda batch: tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        batched=True,
        remove_columns=remove_cols,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def predict_split(
    trainer: Trainer,
    df: pd.DataFrame,
    tokenizer,
    text_column: str,
    label_column: str,
    label_list: list[str],
    max_length: int,
    out_dir: Path,
    split_name: str,
):
    ds = Dataset.from_pandas(df[[text_column]].copy(), preserve_index=False)
    ds = tokenize_unlabeled_dataset(ds, tokenizer, text_column, max_length)

    pred_output = trainer.predict(ds)
    logits = pred_output.predictions
    probs = softmax_np(logits)
    pred_ids = np.argmax(probs, axis=1)
    pred_labels = [label_list[i] for i in pred_ids]
    pred_conf = probs.max(axis=1)

    result = df.copy()
    result["pred_label"] = pred_labels
    result["pred_confidence"] = pred_conf

    pred_csv_path = out_dir / f"{split_name}_predictions.csv"
    result.to_csv(pred_csv_path, index=False, encoding="utf-8")

    metrics = {}
    report_path = out_dir / f"{split_name}_classification_report.json"
    cm_csv_path = out_dir / f"{split_name}_confusion_matrix.csv"
    cm_png_path = out_dir / f"{split_name}_confusion_matrix.png"

    if label_column in df.columns:
        y_true = df[label_column].astype(str).tolist()
        y_pred = pred_labels

        report = classification_report(
            y_true,
            y_pred,
            labels=label_list,
            output_dict=True,
            zero_division=0,
        )
        save_json(report, report_path)
        save_confusion_matrix(y_true, y_pred, label_list, cm_csv_path, cm_png_path)

        metrics = {
            f"{split_name}_accuracy": float(accuracy_score(y_true, y_pred)),
            f"{split_name}_macro_f1": float(f1_score(y_true, y_pred, average='macro')),
            f"{split_name}_weighted_f1": float(f1_score(y_true, y_pred, average='weighted')),
        }

    return {
        "metrics": metrics,
        "pred_csv_path": pred_csv_path,
        "report_path": report_path if report_path.exists() else None,
        "cm_csv_path": cm_csv_path if cm_csv_path.exists() else None,
        "cm_png_path": cm_png_path if cm_png_path.exists() else None,
    }


def main():
    config_path = Path("ml/configs/stage2_baseline_classifier.yaml")
    cfg = load_yaml(config_path)

    experiment_name = cfg["experiment_name"]
    run_name = cfg["run_name"]
    model_name = cfg["model_name"]
    text_column = cfg["text_column"]
    label_column = cfg["label_column"]

    train_path = Path(cfg["paths"]["train"])
    val_path = Path(cfg["paths"]["val"])
    test_path = Path(cfg["paths"]["test"])
    heldout_path = Path(cfg["paths"]["heldout"])

    model_output_dir = Path(cfg["paths"]["model_output_dir"])
    report_dir = Path(cfg["paths"]["report_dir"])
    trainer_output_dir = model_output_dir / "trainer_output"

    model_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    heldout_df = pd.read_csv(heldout_path)

    train_df = ensure_text_column(train_df, text_column)
    val_df = ensure_text_column(val_df, text_column)
    test_df = ensure_text_column(test_df, text_column)
    heldout_df = ensure_text_column(heldout_df, text_column)

    seed = int(cfg["training"]["seed"])
    learning_rate = float(cfg["training"]["learning_rate"])
    batch_size = int(cfg["training"]["batch_size"])
    eval_batch_size = int(cfg["training"]["eval_batch_size"])
    num_train_epochs = int(cfg["training"]["num_train_epochs"])
    weight_decay = float(cfg["training"]["weight_decay"])
    max_length = int(cfg["training"]["max_length"])
    early_stopping_patience = int(cfg["training"]["early_stopping_patience"])
    logging_steps = int(cfg["training"]["logging_steps"])

    set_seed(seed)

    label_list = sorted(train_df[label_column].astype(str).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    label_mapping_path = report_dir / "label_mapping.json"
    save_json(
        {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
        label_mapping_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = build_dataset(train_df, text_column, label_column, label2id)
    val_ds = build_dataset(val_df, text_column, label_column, label2id)
    test_ds = build_dataset(test_df, text_column, label_column, label2id)

    train_ds = tokenize_labeled_dataset(train_ds, tokenizer, text_column, max_length)
    val_ds = tokenize_labeled_dataset(val_ds, tokenizer, text_column, max_length)
    test_ds = tokenize_labeled_dataset(test_ds, tokenizer, text_column, max_length)

    training_args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    with start_run(experiment_name=experiment_name, run_name=run_name):
        log_params(
            {
                "model_name": model_name,
                "text_column": text_column,
                "train_path": str(train_path),
                "val_path": str(val_path),
                "test_path": str(test_path),
                "heldout_path": str(heldout_path),
                "seed": seed,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "max_length": max_length,
                "early_stopping_patience": early_stopping_patience,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "num_labels": len(label_list),
            }
        )

        print("[INFO] Starting training...")
        train_result = trainer.train()
        print("[INFO] Training complete.")

        trainer.save_model(str(model_output_dir))
        tokenizer.save_pretrained(str(model_output_dir))

        val_metrics = trainer.evaluate(eval_dataset=val_ds)

        log_metrics({k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))})

        test_outputs = predict_split(
            trainer=trainer,
            df=test_df,
            tokenizer=tokenizer,
            text_column=text_column,
            label_column=label_column,
            label_list=label_list,
            max_length=max_length,
            out_dir=report_dir,
            split_name="test",
        )
        heldout_outputs = predict_split(
            trainer=trainer,
            df=heldout_df,
            tokenizer=tokenizer,
            text_column=text_column,
            label_column=label_column,
            label_list=label_list,
            max_length=max_length,
            out_dir=report_dir,
            split_name="heldout",
        )

        log_metrics(test_outputs["metrics"])
        log_metrics(heldout_outputs["metrics"])

        save_json(
            {
                "train_metrics": train_result.metrics,
                "val_metrics": val_metrics,
                "test_split_metrics": test_outputs["metrics"],
                "heldout_metrics": heldout_outputs["metrics"],
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
            },
            report_dir / "run_summary.json",
        )

        for path in [
            label_mapping_path,
            report_dir / "run_summary.json",
            test_outputs["pred_csv_path"],
            heldout_outputs["pred_csv_path"],
            test_outputs["report_path"],
            heldout_outputs["report_path"],
            test_outputs["cm_csv_path"],
            heldout_outputs["cm_csv_path"],
            test_outputs["cm_png_path"],
            heldout_outputs["cm_png_path"],
        ]:
            if path is not None and Path(path).exists():
                log_artifact(str(path))

        print("\n[INFO] Best model checkpoint:", trainer.state.best_model_checkpoint)
        print("[INFO] Best val metric:", trainer.state.best_metric)

        print("\n[INFO] Test metrics:")
        print(test_outputs["metrics"])

        print("\n[INFO] Heldout metrics:")
        print(heldout_outputs["metrics"])

        print(f"\n[OK] Model saved to: {model_output_dir}")
        print(f"[OK] Reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
"""
Evaluation: accuracy/precision/recall/specificity/F1/ROC-AUC/confusion matrix,
plus error analysis (false positives, false negatives, low-confidence cases).
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


def evaluate_model(model, val_ds, df_val, name, output_dir="./outputs"):
    y_true = df_val["label"].values
    y_prob = model.predict(val_ds).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall_sensitivity": recall_score(y_true, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm = confusion_matrix(y_true, y_pred)
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{name} - Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{name} - ROC Curve"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures", f"{name}_evaluation.png"), dpi=150)
    plt.close(fig)

    pd.Series(metrics).to_csv(os.path.join(output_dir, "metrics", f"{name}_metrics.csv"))
    return metrics, y_prob, y_pred


def error_analysis(df_val, y_true, y_prob, image_dir, output_dir="./outputs", n_examples=5):
    results = df_val.copy()
    results["y_true"] = y_true
    results["y_prob"] = y_prob
    results["y_pred"] = (y_prob >= 0.5).astype(int)

    false_positives = results[(results.y_true == 0) & (results.y_pred == 1)].sort_values(
        "y_prob", ascending=False
    )
    false_negatives = results[(results.y_true == 1) & (results.y_pred == 0)].sort_values("y_prob")
    low_confidence = results[(results.y_prob > 0.4) & (results.y_prob < 0.6)]

    for title, subset in [
        ("False Positives", false_positives),
        ("False Negatives", false_negatives),
        ("Low-Confidence Predictions", low_confidence),
    ]:
        subset = subset.head(n_examples)
        if len(subset) == 0:
            continue
        fig, axes = plt.subplots(1, len(subset), figsize=(3 * len(subset), 3))
        axes = np.atleast_1d(axes)
        for ax, (_, row) in zip(axes, subset.iterrows()):
            img = cv2.cvtColor(cv2.imread(os.path.join(image_dir, row["id"] + ".tif")), cv2.COLOR_BGR2RGB)
            ax.imshow(img); ax.axis("off")
            ax.set_title(f"true={row.y_true}, p={row.y_prob:.2f}", fontsize=9)
        fig.suptitle(title)
        plt.tight_layout()
        fname = f"error_{title.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, "figures", fname), dpi=150)
        plt.close(fig)

    return false_positives, false_negatives, low_confidence

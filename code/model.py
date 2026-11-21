import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ["Atsargus", "Normalus", "Agresyvus"]

def clean_outliers_iqr(df, column="accel_long_std"):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3.0 * iqr
    cleaned = df[df[column] <= upper_bound].copy()
    print(f"Pašalinta {len(df) - len(cleaned)} jutiklio trikdžių anomalijų (IQR filtras).")
    return cleaned

def train_and_evaluate():
    df = pd.read_csv("data/driving_telemetry.csv")
    df = clean_outliers_iqr(df)

    X = df.drop(columns=["style_label"])
    y = df["style_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Baseline (Logistic Regression)": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Proposed (Gradient Boosting)": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=3, subsample=0.8, random_state=42
        )
    }

    results = {}
    trained_gb = None

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_score": round(float(f1), 4)
        }
        print(f"\n--- {name} ---")
        print(results[name])
        if "Proposed" in name:
            trained_gb = model
            cm = confusion_matrix(y_test, preds)

    os.makedirs("results", exist_ok=True)
    with open("results/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Konfuzijos matrica (Proposed: Gradient Boosting)")

    importances = trained_gb.feature_importances_
    idx = np.argsort(importances)
    axes[1].barh(X.columns[idx], importances[idx], color="#1f4e78")
    axes[1].set_title("XAI: Požymių svarba vairavimo stiliui")
    axes[1].set_xlabel("Santykinė svarba")

    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=200)
    print("\nGrafikas išsaugotas -> results/model_comparison.png")

if __name__ == "__main__":
    train_and_evaluate()

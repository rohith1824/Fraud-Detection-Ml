import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

def precision_at_k(y_true, y_scores, k):
    """
    Compute precision at top k fraction of predictions.
    k: fraction between 0 and 1
    """
    if not 0 < k < 1:
        raise ValueError("k must be between 0 and 1.")
    
    # Number of top samples to consider
    cutoff = max(int(len(y_scores) * k), 1)

    # Indices of top k scores
    idx = np.argsort(y_scores)[-cutoff:]
    return np.sum(y_true[idx]) / cutoff

def main():
    # Evaluation artifacts output directory
    out_dir = os.path.join("reports", "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    # Load model and data
    model = joblib.load(os.path.join("models", "rf_champion.joblib"))
    X_test = np.load(os.path.join("data", "prep", "creditcard_X_test.npy"))
    y_test = np.load(os.path.join("data", "prep", "creditcard_y_test.npy"))

    # Predict probabilities
    y_scores = model.predict_proba(X_test)[:, 1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve (AP = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()

    # Confusion Matrix, threshold=0.3
    y_pred = (y_scores >= 0.3).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Legitimate', 'Predicted Fraudulent'])
    ax.set_yticklabels(['Actual Legitimate', 'Actual Fraudulent'])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center',
                color='white' if val > cm.max()/2 else 'black')
    plt.title("Confusion Matrix (Threshold = 0.3)")
    plt.subplots_adjust(left=0.25)  
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # Numeric summary including Precision@k
    metrics_path = os.path.join(out_dir, "metrics_summary.txt")
    ks = [0.01, 0.05, 0.1]
    with open(metrics_path, "w") as f:
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision (PR AUC): {pr_auc:.4f}\n")
        for k in ks:
            p_at_k = precision_at_k(y_test, y_scores, k)
            f.write(f"Precision@{int(k*100)}%: {p_at_k:.4f}\n")

    print(f"Evaluation complete. Artifacts saved in {out_dir}")


if __name__ == "__main__":
    main()
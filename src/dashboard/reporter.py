import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from config.config_loader import REPORTS_DIR

def generate_roc_curve(model, X_test, y_test, filename='roc_curve.png'):
    """
    Generates and saves the ROC curve plot.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
        filename (str): Name of the file to save the plot to.
    """
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Save the plot to the reports directory
        report_path = os.path.join(REPORTS_DIR, filename) # Use REPORTS_DIR from __init__.py
        plt.savefig(report_path)
        plt.close()  # Close the plot to free memory
        print(f"ROC curve plot saved to {report_path}")

    except Exception as e:
        print(f"An error occurred while generating the ROC curve: {e}")

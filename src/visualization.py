from sklearn.metrics import confusion_matri, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot the confusion matrix for classification tasks.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Custom name to display in the title.
    Usage:
        plot_confusion_matrix(y_test, model.predict(X_test), model_name='SVM Classifier')
    """
    # Generate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Plot confusion matrix using Seaborn heatmap
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    # Labels and title
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16)  # Custom title with model name
    # Display the plot
    plt.show()

def plot_roc_curve(y_true, y_pred, model_name="Model"):
    """
    Plot the ROC curve for binary classification.

    Parameters:
        y_true (array-like): True labels (0 for non-fraud, 1 for fraud).
        y_pred (array-like): Predicted probabilities for the positive class (fraud).
        model_name (str): Custom name to display in the title.

    Usage:
        plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], model_name='SVM Classifier')
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")  # Diagonal line
    plt.title(f'{model_name} - ROC Curve', fontsize=16)  # Custom title with model name
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()
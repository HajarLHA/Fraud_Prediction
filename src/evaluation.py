from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a classification model and print key metrics.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    """
    print("Precision Score:", precision_score(y_true, y_pred))
    print("Recall Score:", recall_score(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

    # AUC score for multi-class requires probabilities
    try:
        auc = roc_auc_score(y_true, y_pred)
        print("AUC Score:", auc)
    except ValueError:
        print("AUC Score: Not applicable (requires probabilities or binary classification)")
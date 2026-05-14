import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from preprocess import load_and_preprocess

def evaluate_all():
    _, X_test, _, y_test, _ = load_and_preprocess('../data/heart.csv')

    model_names = ['svm', 'logistic_regression', 'random_forest', 'xgboost']

    for name in model_names:
        model = joblib.load(f'../models/{name}.pkl')
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"\n{'='*40}")
        print(f"Model: {name.upper()}")
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix — {name}')
        plt.savefig(f'../models/{name}_cm.png')
        plt.clf()

if __name__ == '__main__':
    evaluate_all()
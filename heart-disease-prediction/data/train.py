import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess

def train_all_models():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess('../data/heart.csv')

    models = {
        'svm': SVC(kernel='rbf', probability=True, random_state=42, C=1.0),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,        # ✅ Limit tree depth
        min_samples_split=5, # ✅ Avoid overfitting
        random_state=42
    ),
    'xgboost': XGBClassifier(
        eval_metric='logloss',
        max_depth=3,         # ✅ Limit tree depth
        learning_rate=0.1,   # ✅ Slow learning
        n_estimators=100,
        subsample=0.8,       # ✅ Use 80% data per tree
        random_state=42
    )
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'../models/{name}.pkl')
        trained[name] = model
        print(f"✅ {name} trained and saved.")

    joblib.dump(scaler, '../models/scaler.pkl')

    # ✅ Overfitting Check
    print("\n--- Overfitting Check ---")
    for name, model in trained.items():
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test, test_pred)

        print(f"\n{name}")
        print(f"Train Accuracy : {train_acc:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")

        diff = train_acc - test_acc
        if diff > 0.05:
            print(f"⚠️ Possible Overfitting! Difference: {diff:.4f}")
        else:
            print(f"✅ No Overfitting! Difference: {diff:.4f}")

    return trained, X_test, y_test

if __name__ == '__main__':
    train_all_models()
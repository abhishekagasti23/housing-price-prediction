import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    """Load preprocessed training and testing data"""
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    """Train model and print evaluation metrics"""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R2  : {r2:.4f}")
    print("-" * 30)

    return model, r2


def train_all_models():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
    }

    best_model = None
    best_score = -np.inf
    best_name = None

    for name, model in models.items():
        trained_model, r2 = train_and_evaluate(
            model, name, X_train, y_train, X_test, y_test
        )

        if r2 > best_score:
            best_score = r2
            best_model = trained_model
            best_name = name

    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nBest Model: {best_name}")
    print(f"Best R2 Score: {best_score:.4f}")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    train_all_models()

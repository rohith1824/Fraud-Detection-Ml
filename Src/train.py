import os
import numpy as np
import joblib
from preprocessing import main as preprocess
from sklearn.ensemble import RandomForestClassifier

def main():
    # Run preprocessing to generate train/validation splits
    preprocess()

    # Load data
    X_train = np.load("/Users/rohith/Desktop/fraud-detection-ml/Data/prep/creditcard_X_train.npy")
    y_train = np.load("/Users/rohith/Desktop/fraud-detection-ml/Data/prep/creditcard_y_train.npy")
    X_val   = np.load("/Users/rohith/Desktop/fraud-detection-ml/Data/prep/creditcard_X_val.npy")
    y_val   = np.load("/Users/rohith/Desktop/fraud-detection-ml/Data/prep/creditcard_y_val.npy")

    # Combine train + validation for final training
    X_full = np.vstack((X_train, X_val))
    y_full = np.concatenate((y_train, y_val))

    # Fit the final RandomForest model
    model = RandomForestClassifier(
        n_estimators= 500,
        min_samples_split= 10,
        min_samples_leaf= 2,
        max_features= 0.3,
        max_depth= 10
    )
    model.fit(X_full, y_full)

    # Save the trained model artifact
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "rf_champion.joblib")
    joblib.dump(model, model_path)
    print(f" Model saved to {model_path}")

if __name__ == "__main__":
    main()
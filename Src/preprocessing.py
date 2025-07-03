import numpy as np
import pandas as pd

FEATURES = [
    'V2', 'V3', 'V4',
    'V11', 'V12', 'V14',
    'V16', 'V17', 'V18',
    'V7', 'V10'
]


def load_data(path):
    cols = ['Class'] + FEATURES
    data = pd.read_csv(path, usecols=cols)
    data = data.drop_duplicates()

    return data


def transform_features(data, skew_threshold = 0.75, drop_original = True):
    numeric_feats = data.select_dtypes(include=[np.number]).columns.tolist()

    if 'Class' in numeric_feats:
        numeric_feats.remove('Class')

    # compute skewness
    skew_vals = data[numeric_feats].skew().abs()

    # select features to transform
    to_log = skew_vals[skew_vals > skew_threshold].index.tolist()
    if not to_log:
        return data

    # apply log1p
    for feat in to_log:
        new_col = f"{feat}_log1p"
        data[new_col] = np.log1p(data[feat])
        if drop_original:
            data.drop(columns=[feat], inplace=True)
        print(f"Transformed {feat} (|skew|={skew_vals[feat]:.2f}) → {new_col}")

    return data


def split_and_save(data, save_out):
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Sequential split: first 70% train, next 15% val, last 15% test
    n = len(data)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    # splits
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]

    # save
    np.save(f"{save_out}_X_train.npy", X_train.values)
    np.save(f"{save_out}_y_train.npy", y_train.values)
    np.save(f"{save_out}_X_val.npy",   X_val.values)
    np.save(f"{save_out}_y_val.npy",   y_val.values)
    np.save(f"{save_out}_X_test.npy",  X_test.values)
    np.save(f"{save_out}_y_test.npy",  y_test.values)

def main():
    data_path = "data/creditcard.csv"
    out_prefix = "data/prep/creditcard"

    df = load_data(data_path)
    df = transform_features(df)
    split_and_save(df, out_prefix)


if __name__ == "__main__":
    main()

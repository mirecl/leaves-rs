from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
import pathlib

path = pathlib.Path(__file__).parent.resolve()
FEATURES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

# Initialize data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
x_df = pd.DataFrame(x, columns=FEATURES)
X_train, _, y_train, _ = train_test_split(x_df, y, test_size=0.15)

# Initialize LGBMRegressor
params = {
    "task": "train",
    "boosting": "gbdt",
    "objective": "regression",
    "num_leaves": 10,
    "learnnig_rage": 0.05,
    "verbose": -1,
}
model = LGBMRegressor(**params)

# Fit model
model.fit(X_train, y_train)

# Initialize data
data = {
    "CRIM": [0.82526, 9.51363, 4.0974, 0.6147, 9.82349],
    "ZN": [20.0, 0.0, 0.0, 0.0, 0.0],
    "INDUS": [3.97, 18.1, 19.58, 6.2, 18.1],
    "CHAS": [0.0, 0.0, 0.0, 0.0, 0.0],
    "NOX": [0.647, 0.713, 0.871, 0.507, 0.671],
    "RM": [7.327, 6.728, 5.468, 6.618, 6.794],
    "AGE": [94.5, 94.1, 100.0, 80.8, 98.8],
    "DIS": [2.0788, 2.4961, 1.4118, 3.2721, 1.358],
    "RAD": [5.0, 24.0, 5.0, 8.0, 24.0],
    "TAX": [264.0, 666.0, 403.0, 307.0, 666.0],
    "PTRATIO": [13.0, 20.2, 14.7, 17.4, 20.2],
    "B": [393.42, 6.68, 396.9, 396.9, 396.9],
    "LSTAT": [11.25, 18.71, 26.42, 7.6, 21.24],
}
df = pd.DataFrame(data)

# Get predicted
preds = model.predict(df)
print(f"Preds: {preds}")

# Save model
joblib.dump(model, f"{path}/lightgbm.bin")

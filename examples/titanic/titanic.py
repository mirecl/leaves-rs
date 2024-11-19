import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib
import pathlib
from catboost.datasets import titanic

path = pathlib.Path(__file__).parent.resolve()
CATEGORY_FEATURES = [
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "SibSp",
    "Parch",
    "Ticket",
    "Cabin",
    "Embarked",
]


def set_category_type(X: pd.DataFrame) -> pd.DataFrame:
    for feature in CATEGORY_FEATURES:
        X[feature] = X[feature].astype("category")
    return X


# Initialize data
train_df, _ = titanic()

# statistics
train_df.fillna(-999, inplace=True)

# Separate features and labels
X = train_df.drop("Survived", axis=1)
y = train_df.Survived

# Get non-float type feature index
X = set_category_type(X)

X_train, _, y_train, _ = train_test_split(X, y, train_size=0.75, random_state=42)

# Initialize LGBMClassifier
params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "verbose": -1,
}
model = LGBMClassifier(**params)

# Fit model
model.fit(X_train, y_train)

# Initialize data
data = {
    "PassengerId": ["892", "893", "894", "895", "896"],
    "Pclass": ["3", "3", "2", "3", "3"],
    "Name": [
        "Kelly, Mr. James",
        "Wilkes, Mrs. James (Ellen Needs)",
        "Myles, Mr. Thomas Francis",
        "Wirz, Mr. Albert",
        "Hirvonen, Mrs. Alexander (Helga E Lindqvist)",
    ],
    "Sex": ["male", "female", "male", "male", "female"],
    "Age": [34.5, 47.0, 62.0, 27.0, 22.0],
    "SibSp": ["0", "1", "0", "0", "1"],
    "Parch": ["0", "0", "0", "0", "1"],
    "Ticket": ["330911", "363272", "240276", "315154", "3101298"],
    "Fare": [7.8292, 7.0, 9.687, 8.6625, 12.2875],
    "Cabin": ["-999", "-999", "-999", "-999", "-999"],
    "Embarked": ["Q", "S", "Q", "S", "S"],
}

df = pd.DataFrame(data)
df = set_category_type(df)

# Get predicted Raw
preds_class = model.predict(df)
print(f"Preds `Raw`: {preds_class}")

# Get predicted Probability
preds_proba = model.predict_proba(df)
print(f"Preds `Probability`: {preds_proba}")

# Save model
joblib.dump(model, f"{path}/lightgbm.bin")

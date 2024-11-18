# 🌿 leaves-rs

🌿 <ins>Leaves</ins> is a library implementing prediction code for GBRT (<ins>Gradient Boosting Regression Trees</ins>).\
The goal of the project - make it possible to use models from popular ML-frameworks on **Pure Rust** 🦀.

### 📖 Proposal API for LightGBM

+ Import library:

```rust
use leaves::{LGBMClassifier, LGBMRegressor, LGBMRanker};
```

+ Load models:

```rust
let model_classifier = LGBMClassifier::from_file("model.bin");
let model_regressor = LGBMRegressor::from_file("model.bin");
let model_ranker = LGBMRanker::from_file("model.bin");
```

+ Predict models:

```rust
let features = vec![1.0, 2.0, 3.0];

// Inference `Classifier`.
let preds_classifier = model_classifier.predict(features);
let preds_classifier_proba = model_classifier.predict_proba(features);

// Inference `Regressor`.
let preds_regressor = model_regressor.predict(features);

// Inference `Ranker`.
let preds_ranker = model_ranker.predict(features);
```

### 🤔 Supported framework

+ [ ] LightGBM (<https://github.com/microsoft/LightGBM>)
+ [ ] XGBoost (<https://github.com/dmlc/xgboost>)
+ [ ] CatBoost (<https://github.com/catboost/catboost>)
+ [ ] Scikit-Learn (<https://github.com/scikit-learn/scikit-learn>)

### 👏 Thanks

+ [@dmitryikh](https://github.com/dmitryikh) for <https://github.com/dmitryikh/leaves>

[![Latest version](https://img.shields.io/crates/v/leaves-rs.svg)](https://crates.io/crates/leaves-rs) ![License](https://img.shields.io/crates/l/leaves-rs.svg)

# 🌿 leaves-rs

🌿 <ins>Leaves</ins> is a library implementing prediction code for GBRT (<ins>Gradient Boosting Regression Trees</ins>).\
The goal of the project - make it possible to inference models from popular ML-frameworks on **Pure Rust** 🦀.

### 📝 Install library

Run the following Cargo command in your project directory:

```sh
cargo add leaves-rs
```

Or add the following line to your **Cargo.toml**:

```toml
leaves-rs = "0.0.1"
```

### 📖 Proposal API for LightGBM

+ Import library for exanple `Classifier`:

```rust
use leaves::LGBMClassifier;
```

+ Load models:

```rust
let model = LGBMClassifier::from_file("model.bin");
```

+ Predict models:

```rust
// Create features vector.
let features = vec![1.0, 2.0, 3.0];

// Inference model `Classifier`.
let preds = model.predict(features);
let preds_proba = model.predict_proba(features);
```

> The models `LGBMRegressor` and `LGBMRamker` will be executed in a similar manner.

### 🤔 Supported framework

+ [ ] LightGBM (<https://github.com/microsoft/LightGBM>)
+ [ ] XGBoost (<https://github.com/dmlc/xgboost>)
+ [ ] CatBoost (<https://github.com/catboost/catboost>)
+ [ ] Scikit-Learn (<https://github.com/scikit-learn/scikit-learn>)

### 👏 Thanks

+ [@dmitryikh](https://github.com/dmitryikh) for <https://github.com/dmitryikh/leaves>

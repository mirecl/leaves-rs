# leaves-rs

Pure Rust implementation of prediction part for GBRT (Gradient Boosting Regression Trees) models from popular frameworks 🚀

### Proposal API

+ Import library:

```rust
use leaves::LightGBM;
```

+ Load model:

```rust
let model = LightGBM::from_file("model.bin");
```

+ Predict model:

```rust
let features = vec![1.0, 2.0, 3.0];
let result = model.predict(features);
```

### Supported framework

+ [ ] LightGBM (<https://github.com/microsoft/LightGBM>)
+ [ ] XGBoost (<https://github.com/dmlc/xgboost>)
+ [ ] CatBoost (<https://github.com/catboost/catboost>)

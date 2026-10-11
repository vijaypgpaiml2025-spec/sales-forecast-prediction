import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import mlflow

# Configure MLflow logging to local tracking server inside Actions runner
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Package-Prediction-Experiment")

# Loaded splits downloaded from the preceding data preparation job's artifact
# We now load train.csv and test.csv because we simplified the split artifact outputs
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target_col = "Product_Store_Sales_Total"

Xtrain = train_df.drop(columns=[target_col])
ytrain = train_df[target_col]

Xtest = test_df.drop(columns=[target_col])
ytest = test_df[target_col]

# Define feature groups for SuperKart pipeline
numeric_features = [
    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age'
]
categorical_features = [
    'Product_Sugar_Content', 'Product_Type', 'Store_Size',
    'Store_Location_City_Type', 'Store_Type'
]

# Build preprocessor mapping
preprocessor = make_column_transformer(
    (StandardScaler(), [f for f in numeric_features if f in Xtrain.columns]),
    (OneHotEncoder(handle_unknown='ignore'), [f for f in categorical_features if f in Xtrain.columns])
)

# Define regressors and parameters grid for continuous predictions
xb_model = xgb.XGBRegressor(random_state=42)
param_grid = {
    'xgbregressor__n_estimators': [50, 70, 100],
    'xgbregressor__max_depth': [3, 4, 5],
    'xgbregressor__learning_rate': [0.05, 0.1]
}

model_pipeline = make_pipeline(preprocessor, xb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='r2')
    grid_search.fit(Xtrain, ytrain)

    # Log trial parameters and scoring metrics to MLflow
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Run inference metrics
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    train_rmse = np.sqrt(mean_squared_error(ytrain, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(ytest, y_pred_test))
    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)

    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

    # Create deploy folder and save model
    os.makedirs("SuperKart_project/deployment", exist_ok=True)
    model_path = "SuperKart_project/deployment/best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model serialized and saved successfully to {model_path}")

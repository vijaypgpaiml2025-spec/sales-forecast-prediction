import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # 1. Load dataset dynamically relative to the script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Since this file sits in SuperKart_project/model_building/,
    # the root project directory is one level up
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    data_path = os.path.join(project_root, "data", "SuperKart.csv")

    # Fallback to local working directory path if relative search fails
    if not os.path.exists(data_path):
        data_path = "SuperKart_project/data/SuperKart.csv"

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")

    # 2. Data Cleaning & Standardization
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    df = df.dropna(subset=["Product_Store_Sales_Total"])

    # 3. Feature Engineering
    if "Store_Establishment_Year" in df.columns:
        df["Store_Age"] = 2026 - df["Store_Establishment_Year"]
        df = df.drop(columns=["Store_Establishment_Year"])

    columns_to_drop = ["Product_Id", "Store_Id"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 4. Categorical Encoding (One-Hot Encoding)
    categorical_cols = [
        "Product_Sugar_Content",
        "Product_Type",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type"
    ]
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)

    # 5. Define Features (X) and Target (y)
    target = "Product_Store_Sales_Total"
    X = df.drop(columns=[target])
    y = df[target]

    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = X_train.copy()
    train_data[target] = y_train

    test_data = X_test.copy()
    test_data[target] = y_test

    # Save outputs to the runner working directory root so train.py and GHA upload-artifact find them seamlessly
    train_data.to_csv("train.csv", index=False)
    test_data.to_csv("test.csv", index=False)

    # Save copies inside structured dataset folders
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_data.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"Data preparation completed successfully!")

if __name__ == "__main__":
    prepare_data()

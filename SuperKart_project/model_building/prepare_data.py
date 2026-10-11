import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # 1. Load dataset from the data folder
    data_path = "data/SuperKart.csv"
    if not os.path.exists(data_path):
        # Fallback check if named sales_data.csv
        data_path = "data/sales_data.csv"

    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")

    # 2. Data Cleaning & Standardization
    # Standardize 'Product_Sugar_Content' (fix 'reg' -> 'Regular')
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    # Handle missing values if any
    df = df.dropna(subset=["Product_Store_Sales_Total"])

    # 3. Feature Engineering
    # Calculate Store Age based on establishment year (Current Year: 2026)
    if "Store_Establishment_Year" in df.columns:
        df["Store_Age"] = 2026 - df["Store_Establishment_Year"]
        df = df.drop(columns=["Store_Establishment_Year"])

    # 4. Drop unnecessary high-cardinality ID columns if they don't contribute directly
    # (Keeping or dropping IDs: Product_Id and Store_Id can be dropped for standard tabular ML regressors)
    columns_to_drop = ["Product_Id", "Store_Id"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 5. Categorical Encoding (One-Hot Encoding)
    categorical_cols = [
        "Product_Sugar_Content",
        "Product_Type",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type"
    ]
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)

    # 6. Define Features (X) and Target (y)
    target = "Product_Store_Sales_Total"
    X = df.drop(columns=[target])
    y = df[target]

    # 7. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine back to save cleaned files locally
    train_data = X_train.copy()
    train_data[target] = y_train

    test_data = X_test.copy()
    test_data[target] = y_test

    os.makedirs("data/processed", exist_ok=True)
    train_data.to_csv("data/processed/train.csv", index=False)
    test_data.to_csv("data/processed/test.csv", index=False)

    print(f"Data preparation completed successfully!")
    print(f"Training set shape: {train_data.shape}")
    print(f"Testing set shape: {test_data.shape}")
    print(f"Saved processed files to data/processed/train.csv and data/processed/test.csv")

if __name__ == "__main__":
    prepare_data()

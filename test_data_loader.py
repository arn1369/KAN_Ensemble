
from data_loader import prepare_data
import numpy as np

def test_prepare_data():
    print("Testing prepare_data...")
    try:
        X_train, X_test, y_train, y_test, dates_test, feature_cols, scaler_y = prepare_data(
            ticker="GOOGL", 
            start_date="2023-01-01", 
            end_date="2024-01-01", 
            lookback=30,
            split_percent=0.8
        )
        print("Success!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"dates_test length: {len(dates_test)}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prepare_data()

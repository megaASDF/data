import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from preprocessing import preprocess_data


def train_model():
    """
    Train an XGBoost model to predict Avg_Trans_Amount.
    """
    print("=" * 60)
    print("Training XGBoost Model for Avg_Trans_Amount Prediction")
    print("=" * 60)
    
    # Load training data
    print("\n1. Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    print(f"   - Loaded {len(train_df)} rows")
    print(f"   - Columns: {train_df.columns.tolist()}")
    
    # Separate features and target
    print("\n2. Preparing features and target...")
    y = train_df['Avg_Trans_Amount'].copy()
    X_raw = train_df.drop(columns=['Avg_Trans_Amount'])
    
    # Preprocess features
    print("\n3. Preprocessing features...")
    X, scaler = preprocess_data(X_raw, fit_scaler=True)
    print(f"   - Feature shape: {X.shape}")
    print(f"   - Features: {X.columns.tolist()}")
    print("   - Features scaled using StandardScaler")
    
    # Split data for validation
    print("\n4. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   - Training set: {len(X_train)} rows")
    print(f"   - Validation set: {len(X_val)} rows")
    
    # Train model
    print("\n5. Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   - Model trained successfully!")
    
    # Evaluate on training set
    print("\n6. Evaluating model...")
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"   Training RMSE: {train_rmse:.4f}")
    print(f"   Training MAE:  {train_mae:.4f}")
    print(f"   Training R²:   {train_r2:.4f}")
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"   Validation RMSE: {val_rmse:.4f}")
    print(f"   Validation MAE:  {val_mae:.4f}")
    print(f"   Validation R²:   {val_r2:.4f}")
    
    # Load and test on test.csv
    print("\n7. Testing on test.csv...")
    test_df = pd.read_csv('data/test.csv')
    print(f"   - Loaded {len(test_df)} test rows")
    
    # Check if test set has target variable
    has_target = 'Avg_Trans_Amount' in test_df.columns
    
    if has_target:
        y_test = test_df['Avg_Trans_Amount'].copy()
        X_test_raw = test_df.drop(columns=['Avg_Trans_Amount'])
    else:
        y_test = None
        X_test_raw = test_df.copy()
    
    # Preprocess test features (using fitted scaler)
    X_test, _ = preprocess_data(X_test_raw, scaler=scaler, fit_scaler=False)
    print(f"   - Test feature shape: {X_test.shape}")
    
    # Make predictions on test set
    y_test_pred = model.predict(X_test)
    
    # Evaluate if target is available
    if has_target:
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"   Test RMSE: {test_rmse:.4f}")
        print(f"   Test MAE:  {test_mae:.4f}")
        print(f"   Test R²:   {test_r2:.4f}")
    else:
        print("   No target variable in test set - predictions generated only")
        print(f"   Prediction stats:")
        print(f"   - Mean: {y_test_pred.mean():.4f}")
        print(f"   - Std:  {y_test_pred.std():.4f}")
        print(f"   - Min:  {y_test_pred.min():.4f}")
        print(f"   - Max:  {y_test_pred.max():.4f}")
    
    # Save predictions
    print("\n8. Saving predictions...")
    predictions_df = pd.DataFrame({
        'prediction': y_test_pred
    })
    if 'Customer_number' in test_df.columns:
        predictions_df.insert(0, 'Customer_number', test_df['Customer_number'])
    
    predictions_path = 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   - Predictions saved to: {predictions_path}")
    
    # Save model and scaler
    print("\n9. Saving model and scaler...")
    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/xgb_model.joblib'
    scaler_path = 'saved_models/scaler.joblib'
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"   - Model saved to: {model_path}")
    print(f"   - Scaler saved to: {scaler_path}")
    
    print("\n" + "=" * 60)
    print("Training and testing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Change to src directory to import preprocessing
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')  # Go back to project root
    
    train_model()

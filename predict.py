import os
import pandas as pd
import joblib
import sys
from src.preprocessing import preprocess_data

# ============================================================
# ❌ DO NOT MODIFY
# These paths are fixed for competition evaluation.
# The grading server will mount data to these locations.
# ============================================================
INPUT_PATH = os.environ.get('INPUT_PATH', '/data/input')
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/data/output')
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "model.joblib")


# ============================================================
# ❌ DO NOT MODIFY
# System path validation functions
# ============================================================
def validate_paths():
    """Validate input/output paths exist and create output directory if needed."""
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input directory not found: {INPUT_PATH}")
        print("Please ensure input data is mounted to /data/input")
        sys.exit(1)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)


def get_input_files():
    """Get list of CSV files to process from INPUT_PATH."""
    files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".csv")]
    if not files:
        print(f"WARNING: No CSV files found in {INPUT_PATH}")
        print("Inference completed with no files to process.")
    return files


# ============================================================
# ✅ MODIFIABLE
# Model loading - Customize based on your model type
# ============================================================
def load_model():
    """
    Load the trained model from MODEL_PATH.
    
    ✅ MODIFIABLE:
    - Change how to load model (joblib, torch, tensorflow, etc.)
    - Load additional artifacts (scaler, encoder, etc.)
    - Load ensemble models
    
    Returns:
        model: The loaded model object
    """
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Please ensure your model is saved in saved_models/model.joblib")
        sys.exit(1)
    
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


# ============================================================
# ✅ MODIFIABLE
# Data preprocessing and prediction logic
# ============================================================
def preprocess_input(df):
    """
    Preprocess input data before prediction.
    
    ✅ MODIFIABLE:
    - Change preprocessing logic
    - Add feature engineering
    - Handle missing values differently
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        X: Preprocessed features ready for prediction
        customer_number: Customer_number column to include in output
    """
    # Keep Customer_number column
    customer_number = None
    if 'Customer_number' in df.columns:
        customer_number = df['Customer_number'].copy()
        print(f"  - Found 'Customer_number' column")
    
    # Preprocess data
    print(f"  - Preprocessing...")
    X = preprocess_data(df.copy())
    
    return X, customer_number


def make_predictions(model, X):
    """
    Make predictions using the loaded model.
    
    ✅ MODIFIABLE:
    - Change prediction logic (ensemble, averaging, etc.)
    - Post-process predictions
    - Apply business rules/constraints
    
    Args:
        model: The loaded model
        X: Preprocessed features
        
    Returns:
        predictions: Array of predictions
    """
    print(f"  - Predicting...")
    predictions = model.predict(X)
    
    # ✅ You can add post-processing here
    # Example: predictions = np.clip(predictions, 0, None)  # Ensure non-negative
    
    return predictions


# ============================================================
# ❌ DO NOT MODIFY
# Output format must match competition requirements
# ============================================================
def save_predictions(predictions, customer_number):
    """
    Save predictions to output file with required format.
    
    ❌ DO NOT MODIFY:
    - Output format must be: Customer_number, Avg_Trans_Amount
    - File naming: submission.csv
    """
    result_df = pd.DataFrame()
    if customer_number is not None:
        result_df['Customer_number'] = customer_number
    
    result_df['Avg_Trans_Amount'] = predictions
    
    output_file = os.path.join(OUTPUT_PATH, f"submission.csv")
    result_df.to_csv(output_file, index=False)
    print(f"  - Saved predictions to: submission.csv")


def process_single_file(model, filename):
    """
    Process a single input file: load, preprocess, predict, save.
    
    ⚠️ LIGHTLY MODIFIABLE:
    - Can add error handling
    - Can add more detailed logging
    """
    file_path = os.path.join(INPUT_PATH, filename)
    try:
        print(f"Processing: {filename}")
        df = pd.read_csv(file_path)
        print(f"  - Loaded {len(df)} rows")
        
        # ✅ Modifiable: Preprocessing
        X, customer_number = preprocess_input(df)
        
        # ✅ Modifiable: Prediction
        predictions = make_predictions(model, X)
        
        # ❌ Fixed: Save output
        save_predictions(predictions, customer_number)
        
        print(f"  ✓ Success\n")
        
    except Exception as e:
        print(f"  ✗ ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        print()


# ============================================================
# ❌ DO NOT MODIFY
# Main inference pipeline
# ============================================================
def run_inference():
    
    # Validate system paths
    validate_paths()
    
    # Load model
    model = load_model()
    
    # Get input files
    files = get_input_files()
    if not files:
        return
    
    print(f"\nFound {len(files)} CSV file(s) to process: {files}\n")
    
    # Process each file
    for filename in files:
        process_single_file(model, filename)
    
    print("=" * 60)
    print("Inference completed.")
    print("=" * 60)

if __name__ == "__main__":
    run_inference()

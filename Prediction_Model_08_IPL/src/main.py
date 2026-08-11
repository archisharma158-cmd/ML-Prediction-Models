#!/usr/bin/env python3
"""
IPL 1st Inning Score Prediction - Main Pipeline

This script orchestrates the entire machine learning pipeline:
1. Load Data       - Load the IPL ball-by-ball dataset from data/players.xlsx
2. Preprocess      - Drop irrelevant columns, filter teams, remove first 5 overs
3. Feature Engineer - Label encode, one-hot encode, build feature DataFrame
4. Train Models    - Train 6 regression models (Decision Tree, Linear Regression,
                     Random Forest, Lasso, SVR, Neural Network)
5. Evaluate Models - Compute R², MAE, MSE, RMSE; generate correlation heatmap
                     and model comparison barplot; save metrics
6. Select Best Model - Random Forest (best performer from notebook analysis)
7. Save Models     - Save top 3 models (Random Forest, Decision Tree, Neural Network)
                     using joblib.dump()
8. Run Predictions - Execute 6 test cases and save results to CSV

Usage:
    python main.py              # Run the full pipeline
    python main.py --skip-train # Skip training if models already exist

Requirements:
    - The dataset must be placed at data/players.xlsx
    - Required columns: mid, date, venue, batting_team, bowling_team, batsman,
      bowler, runs, wickets, overs, runs_last_5, wickets_last_5, striker,
      non-striker, total
"""

import os
import sys
import argparse
from joblib import dump

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, display_dataset_info
from src.preprocessing import (
    drop_irrelevant_columns,
    filter_consistent_teams,
    remove_first_5_overs,
)
from src.feature_engineering import build_features
from src.train import prepare_train_test_split, train_all_models
from src.evaluate import (
    evaluate_all_models,
    plot_correlation_matrix,
    plot_model_comparison,
    save_metrics,
    select_best_model,
)
from src.predict import run_all_tests, save_predictions


def parse_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="IPL 1st Inning Score Prediction Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/players.xlsx",
        help="Path to the IPL dataset Excel file (default: data/players.xlsx)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training if models already exist",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save/load models (default: models)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory for outputs (default: outputs)",
    )
    return parser.parse_args()


def ensure_directories(dirs):
    """
    Create required directories if they don't exist.

    Parameters
    ----------
    dirs : list
        List of directory paths to create.
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def main():
    """
    Execute the complete IPL score prediction pipeline.

    The pipeline performs the following steps in order:
    1. Parse command-line arguments
    2. Create required directories
    3. Load the dataset
    4. Display basic dataset information (EDA)
    5. Preprocess the data (clean, filter, transform)
    6. Build features (encode, transform)
    7. Split into training and testing sets
    8. Train all 6 regression models
    9. Evaluate all models and save metrics
    10. Generate and save visualizations
    11. Select and save the best model
    12. Run test predictions and save results
    """
    # Parse arguments
    args = parse_arguments()

    # Ensure required directories exist
    ensure_directories([
        args.models_dir,
        os.path.join(args.outputs_dir, "plots"),
    ])

    print("=" * 70)
    print("🏏 IPL 1st INNING SCORE PREDICTION PIPELINE")
    print("=" * 70)

    # =========================================================
    # STEP 1: Load Data
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    if not os.path.exists(args.data_path):
        print(f"\n❌ ERROR: Dataset not found at '{args.data_path}'")
        print("Please place the IPL ball-by-ball dataset as data/players.xlsx")
        print("Expected columns: mid, date, venue, batting_team, bowling_team,")
        print("batsman, bowler, runs, wickets, overs, runs_last_5,")
        print("wickets_last_5, striker, non-striker, total")
        sys.exit(1)

    df = load_data(args.data_path)
    print(f"\n✅ Dataset loaded successfully!")
    display_dataset_info(df)

    # =========================================================
    # STEP 2: Data Preprocessing
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)

    df = drop_irrelevant_columns(df)
    df = filter_consistent_teams(df)
    df = remove_first_5_overs(df)
    print("\n✅ Preprocessing complete!")

    # =========================================================
    # STEP 3: Feature Engineering
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)

    df_features = build_features(df)
    print("\n✅ Feature engineering complete!")
    print(f"Final feature set: {df_features.shape[1] - 1} features + 1 target")

    # =========================================================
    # STEP 4: Train-Test Split
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("=" * 70)

    train_features, test_features, train_labels, test_labels = prepare_train_test_split(
        df_features
    )
    print("\n✅ Train-test split complete!")

    # =========================================================
    # STEP 5: Train Models
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING")
    print("=" * 70)

    if args.skip_train:
        print("\n⏭️ Skipping training (--skip-train flag detected)")
        print("Loading pre-trained models from disk...")
        from joblib import load
        models = {}
        for model_name in ["tree", "linreg", "forest", "lasso", "svm", "neural_net"]:
            model_path = os.path.join(args.models_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                models[model_name] = load(model_path)
                print(f"  Loaded {model_name} from {model_path}")
            else:
                print(f"  ⚠️ {model_name} not found at {model_path}")
    else:
        models = train_all_models(train_features, train_labels)

    print("\n✅ Model training complete!")

    # =========================================================
    # STEP 6: Evaluate Models
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 70)

    results, models_tracker = evaluate_all_models(
        models, train_features, train_labels, test_features, test_labels
    )

    # Save metrics to file
    metrics_path = os.path.join(args.outputs_dir, "metrics.txt")
    save_metrics(results, save_path=metrics_path)

    # Generate and save correlation matrix plot
    corr_path = os.path.join(args.outputs_dir, "plots", "correlation_matrix.png")
    plot_correlation_matrix(df_features, save_path=corr_path)

    # Generate and save model comparison plot
    comp_path = os.path.join(args.outputs_dir, "plots", "model_comparison.png")
    plot_model_comparison(models_tracker, save_path=comp_path)

    # Check metric logs
    if results.get("forest"):
        forest_result = results["forest"]
        print(f"\n📊 Random Forest Performance:")
        print(f"   Test R² Score: {forest_result['test_score']}%")
        print(f"   RMSE: {forest_result['rmse']:.4f}")
        print(f"   MAE: {forest_result['mae']:.4f}")

    print("\n✅ Evaluation complete!")

    # =========================================================
    # STEP 7: Select and Save Best Model
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 7: SAVING MODELS")
    print("=" * 70)

    # Save all models
    for model_name, model in models.items():
        model_path = os.path.join(args.models_dir, f"{model_name}_model.pkl")
        # Use consistent naming: forest_model, tree_model, neural_nets_model
        if model_name == "forest":
            model_path = os.path.join(args.models_dir, "forest_model.pkl")
        elif model_name == "tree":
            model_path = os.path.join(args.models_dir, "tree_model.pkl")
        elif model_name == "neural_net":
            model_path = os.path.join(args.models_dir, "neural_nets_model.pkl")
        else:
            model_path = os.path.join(args.models_dir, f"{model_name}_model.pkl")

        dump(model, model_path)
        print(f"  ✅ Saved: {model_path}")

    # Select best model (Random Forest)
    best_model = select_best_model(models)

    # Also save best model explicitly
    best_model_path = os.path.join(args.models_dir, "trained_model.pkl")
    dump(best_model, best_model_path)
    print(f"  ✅ Best model (Random Forest) saved as: {best_model_path}")

    print("\n✅ Models saved successfully!")

    # =========================================================
    # STEP 8: Run Predictions
    # =========================================================
    print("\n" + "=" * 70)
    print("STEP 8: RUNNING PREDICTIONS")
    print("=" * 70)

    predictions_df = run_all_tests(best_model)
    predictions_path = os.path.join(args.outputs_dir, "predictions.csv")
    save_predictions(predictions_df, save_path=predictions_path)

    print("\n✅ Predictions complete!")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("🎯 PIPELINE EXECUTION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Outputs Generated:")
    print(f"   - Models saved in:       {args.models_dir}/")
    print(f"   - Evaluation Metrics:    {metrics_path}")
    print(f"   - Correlation Plot:      {corr_path}")
    print(f"   - Model Comparison Plot:  {comp_path}")
    print(f"   - Predictions CSV:        {predictions_path}")
    print(f"\n🚀 To make a custom prediction, run:")
    print(f"   python -c \"from src.predict import *; print(predict_score('Team A', 'Team B', overs=10, runs=50, wickets=2, runs_last_5=30, wickets_last_5=1, model=load_model('models/forest_model.pkl')))\"")
    print("=" * 70)


if __name__ == "__main__":
    main()


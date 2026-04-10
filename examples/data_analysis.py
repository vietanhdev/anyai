"""Auto ML pipeline: load CSV, profile, clean, train, explain.

Usage: python data_analysis.py dataset.csv target_column

Install: pip install anyai[table,ml]
"""
import sys

import pandas as pd


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    target = sys.argv[2] if len(sys.argv) > 2 else "label"

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")

    # Step 1: Profile the data
    import tableai
    prof = tableai.profile(df)
    print(f"\n--- Data Profile ---")
    print(f"Rows: {prof.n_rows}, Columns: {prof.n_cols}")
    print(f"Missing: {prof.missing_total} values")

    # Step 2: Clean the data
    cleaned = tableai.clean(df)
    print(f"\n--- Cleaned ---")
    print(f"Shape after cleaning: {cleaned.data.shape}")

    # Step 3: Auto-train a classifier
    import anyml
    result = anyml.classify(cleaned.data, target=target, progress=True)
    print(f"\n--- Training Results ---")
    print(f"Best model: {result.model_name} (score: {result.score:.4f})")
    print(f"All models: {result.all_scores}")

    # Step 4: Explain the model
    print(f"\n--- Feature Importance ---")
    for feat, imp in list(result.feature_importances.items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    print(f"\n{result.report()}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Demo mode: provide a CSV file to analyze.")
        print("Usage: python data_analysis.py dataset.csv target_column")
        print("Install: pip install anyai[table,ml]")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install anyai[table,ml]")

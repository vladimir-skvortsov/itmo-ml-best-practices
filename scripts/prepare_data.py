"""
Prepare and split data for ML pipeline.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(
    input_file: str,
    train_output: str,
    test_output: str,
    metadata_output: str,
    test_size: float,
    random_state: int,
    target_col: str,
) -> None:
    """Prepare and split data."""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Remove Id column if exists
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_col}")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if categorical
    label_encoder = None
    if y.dtype == object or isinstance(y.iloc[0], str):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        classes = label_encoder.classes_.tolist()
        print(f"Encoded {len(classes)} classes: {classes}")
    else:
        y_encoded = y.values
        classes = None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Save splits
    train_df = X_train.copy()
    train_df[target_col] = y_train
    train_df.to_csv(train_output, index=False)

    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_df.to_csv(test_output, index=False)

    # Save metadata
    metadata = {
        "target_col": target_col,
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0],
        "test_size": test_size,
        "random_state": random_state,
        "classes": classes,
    }

    Path(metadata_output).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved train data to {train_output}")
    print(f"Saved test data to {test_output}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    # Snakemake variables (snakemake object is injected by Snakemake at runtime)
    prepare_data(
        input_file=snakemake.input.raw_data,  # type: ignore  # noqa: F821
        train_output=snakemake.output.train_data,  # type: ignore  # noqa: F821
        test_output=snakemake.output.test_data,  # type: ignore  # noqa: F821
        metadata_output=snakemake.output.metadata,  # type: ignore  # noqa: F821
        test_size=snakemake.params.test_size,  # type: ignore  # noqa: F821
        random_state=snakemake.params.random_state,  # type: ignore  # noqa: F821
        target_col=snakemake.params.target_col,  # type: ignore  # noqa: F821
    )

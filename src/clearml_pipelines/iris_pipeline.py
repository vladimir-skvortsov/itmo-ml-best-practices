from clearml import Task
from clearml.automation import PipelineController


def create_iris_pipeline() -> None:
    """Create and execute ClearML pipeline for Iris classification."""

    # Create pipeline controller
    pipeline = PipelineController(
        project="iris-pipeline",
        name="iris-classification-pipeline",
        version="1.0",
        add_pipeline_tags=True,
    )

    pipeline.set_default_execution_queue("default")

    # Step 1: Data Preparation
    pipeline.add_step(
        name="data_preparation",
        base_task_project="iris-pipeline",
        base_task_name="prepare_data",
        parameter_override={
            "General/data_path": "data/raw/iris.csv",
            "General/test_size": 0.2,
            "General/random_state": 42,
        },
    )

    # Step 2: Train Logistic Regression
    pipeline.add_step(
        name="train_logistic_regression",
        parents=["data_preparation"],
        base_task_project="iris-pipeline",
        base_task_name="train_model",
        parameter_override={
            "General/model_type": "logistic_regression",
            "General/model_params": {"max_iter": 1000, "random_state": 42},
        },
    )

    # Step 3: Train Random Forest
    pipeline.add_step(
        name="train_random_forest",
        parents=["data_preparation"],
        base_task_project="iris-pipeline",
        base_task_name="train_model",
        parameter_override={
            "General/model_type": "random_forest",
            "General/model_params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
        },
    )

    # Step 4: Train Gradient Boosting
    pipeline.add_step(
        name="train_gradient_boosting",
        parents=["data_preparation"],
        base_task_project="iris-pipeline",
        base_task_name="train_model",
        parameter_override={
            "General/model_type": "gradient_boosting",
            "General/model_params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
        },
    )

    # Step 5: Train SVM
    pipeline.add_step(
        name="train_svm",
        parents=["data_preparation"],
        base_task_project="iris-pipeline",
        base_task_name="train_model",
        parameter_override={
            "General/model_type": "svm",
            "General/model_params": {"kernel": "rbf", "C": 1.0, "random_state": 42},
        },
    )

    # Step 6: Compare Models
    pipeline.add_step(
        name="compare_models",
        parents=[
            "train_logistic_regression",
            "train_random_forest",
            "train_gradient_boosting",
            "train_svm",
        ],
        base_task_project="iris-pipeline",
        base_task_name="compare_models",
    )

    # Step 7: Deploy Best Model
    pipeline.add_step(
        name="deploy_best_model",
        parents=["compare_models"],
        base_task_project="iris-pipeline",
        base_task_name="deploy_model",
    )

    print("Pipeline structure created:")
    print("1. data_preparation")
    print("2. train_logistic_regression (depends on 1)")
    print("3. train_random_forest (depends on 1)")
    print("4. train_gradient_boosting (depends on 1)")
    print("5. train_svm (depends on 1)")
    print("6. compare_models (depends on 2,3,4,5)")
    print("7. deploy_best_model (depends on 6)")

    # Start the pipeline
    print("\nStarting pipeline execution...")
    pipeline.start()

    # Wait for pipeline to complete
    print("Waiting for pipeline to complete...")
    pipeline.wait()

    # Check pipeline status
    print(f"\nPipeline completed with status: {pipeline.status}")
    print("View pipeline in ClearML Web UI: http://localhost:8080")


def create_pipeline_tasks() -> None:
    """Create template tasks for the pipeline."""

    # Task 1: Data Preparation
    task = Task.create(
        project_name="iris-pipeline",
        task_name="prepare_data",
        task_type=Task.TaskTypes.data_processing,
        script="""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from clearml import Task

task = Task.current_task()
params = {
    "data_path": "data/raw/iris.csv",
    "test_size": 0.2,
    "random_state": 42,
}
task.connect(params)

# Load data
df = pd.read_csv(params["data_path"])
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Encode target
target_col = "Species"
if df[target_col].dtype == "object":
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

# Split data
X = df.drop(columns=[target_col]).values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y
)

# Save splits
task.upload_artifact("X_train", X_train)
task.upload_artifact("X_test", X_test)
task.upload_artifact("y_train", y_train)
task.upload_artifact("y_test", y_test)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
""",
    )
    task.close()
    print("Created task: prepare_data")

    # Task 2: Train Model (template)
    task = Task.create(
        project_name="iris-pipeline",
        task_name="train_model",
        task_type=Task.TaskTypes.training,
        script="""
from clearml import Task
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

task = Task.current_task()
params = {
    "model_type": "random_forest",
    "model_params": {},
}
task.connect(params)

# Get data from previous task
parent_task = task.get_task(
    project_name="iris-pipeline",
    task_name="prepare_data"
)
X_train = parent_task.artifacts["X_train"].get()
X_test = parent_task.artifacts["X_test"].get()
y_train = parent_task.artifacts["y_train"].get()
y_test = parent_task.artifacts["y_test"].get()

# Create model
models = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
}
model = models[params["model_type"]](**params["model_params"])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

task.get_logger().report_single_value("test_accuracy", accuracy)
task.get_logger().report_single_value("test_f1", f1)

# Save model
model_path = f"models/pipeline/{params['model_type']}_model.pkl"
joblib.dump(model, model_path)
task.upload_artifact("model", model_path)

print(f"Model: {params['model_type']}, Accuracy: {accuracy:.4f}")
""",
    )
    task.close()
    print("Created task: train_model")

    # Task 3: Compare Models
    task = Task.create(
        project_name="iris-pipeline",
        task_name="compare_models",
        task_type=Task.TaskTypes.qc,
        script="""
from clearml import Task
import pandas as pd

task = Task.current_task()

# Get all training tasks
project = Task.get_projects(["iris-pipeline"])[0]
training_tasks = Task.get_tasks(
    project_name="iris-pipeline",
    task_name="train_model*"
)

# Collect metrics
results = []
for train_task in training_tasks:
    metrics = train_task.get_last_scalar_metrics()
    results.append({
        "model": train_task.get_parameters()["General/model_type"],
        "accuracy": metrics["test_accuracy"]["value"],
        "f1": metrics["test_f1"]["value"],
    })

# Create comparison
df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print("Model Comparison:")
print(df.to_string(index=False))

# Save best model reference
best_model = df.iloc[0]["model"]
task.set_parameter("best_model", best_model)
task.upload_artifact("comparison", df)
""",
    )
    task.close()
    print("Created task: compare_models")

    # Task 4: Deploy Model
    task = Task.create(
        project_name="iris-pipeline",
        task_name="deploy_model",
        task_type=Task.TaskTypes.service,
        script="""
from clearml import Task

task = Task.current_task()

# Get best model from comparison
comparison_task = task.get_task(
    project_name="iris-pipeline",
    task_name="compare_models"
)
best_model_type = comparison_task.get_parameters()["General/best_model"]

print(f"Deploying best model: {best_model_type}")

# In production, this would deploy the model to a serving endpoint
# For now, just log the deployment
task.get_logger().report_text(f"Model {best_model_type} deployed successfully")
""",
    )
    task.close()
    print("Created task: deploy_model")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("ClearML Pipeline Setup")
    print("=" * 60)

    # Create template tasks
    print("\nCreating pipeline template tasks...")
    create_pipeline_tasks()

    print("\nTemplate tasks created successfully!")
    print("\nTo run the pipeline:")
    print("1. Start ClearML Server: make clearml-up")
    print("2. Run this script again")
    print("3. View pipeline in Web UI: http://localhost:8080")

    # Note: Uncomment below to actually run the pipeline
    # print("\nCreating and executing pipeline...")
    # create_iris_pipeline()


if __name__ == "__main__":
    main()

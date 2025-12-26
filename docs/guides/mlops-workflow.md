# ClearML

## Setup

```bash
make clearml-up  # http://localhost:8080

# Get credentials from Settings
cp clearml.conf.example clearml.conf
# Add credentials
cp clearml.conf ~/clearml.conf
```

**Never commit `clearml.conf`!**

## Usage

```python
from clearml import Task

task = Task.init(project_name="iris", task_name="rf")
model.fit(X_train, y_train)
task.get_logger().report_single_value("accuracy", accuracy)
task.upload_artifact("model", "model.pkl")
task.close()
```

## Commands

```bash
make clearml-train       # Single model
make clearml-experiment  # 17 experiments
make clearml-pipeline    # Full pipeline
```

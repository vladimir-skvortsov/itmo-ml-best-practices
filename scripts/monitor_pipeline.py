"""
Monitor Snakemake pipeline execution and send notifications.
"""

from pathlib import Path

import click


def send_notification(message: str, status: str = "info") -> None:
    """Send notification (placeholder for actual notification system)."""
    prefix = {
        "info": "[INFO]",
        "success": "[SUCCESS]",
        "error": "[ERROR]",
        "warning": "[WARNING]",
    }
    print(f"\n{prefix.get(status, '')} {message}\n")


def monitor_pipeline(pipeline_dir: str = ".") -> None:
    """Monitor pipeline execution and generate report."""
    pipeline_path = Path(pipeline_dir)

    # Check for log files
    logs_dir = pipeline_path / "reports" / "logs"
    if not logs_dir.exists():
        send_notification("No logs directory found", "warning")
        return

    log_files = list(logs_dir.glob("*.log"))

    if not log_files:
        send_notification("No log files found", "warning")
        return

    print("=" * 60)
    print("Pipeline Execution Monitor")
    print("=" * 60)
    print(f"Logs directory: {logs_dir}")
    print(f"Total log files: {len(log_files)}\n")

    # Analyze each log
    for log_file in sorted(log_files):
        print(f"[LOG] {log_file.name}")
        try:
            with open(log_file) as f:
                content = f.read()
                lines = len(content.split("\n"))
                print(f"   Lines: {lines}")

                if "error" in content.lower():
                    print("   [ERROR] Errors detected")
                elif "warning" in content.lower():
                    print("   [WARNING] Warnings found")
                else:
                    print("   [OK] Completed successfully")
        except Exception as e:
            print(f"   [ERROR] Error reading file: {e}")
        print()

    # Check for output files
    models_dir = pipeline_path / "models"
    reports_dir = pipeline_path / "reports"

    if models_dir.exists():
        model_files = list(models_dir.glob("*/model.pkl"))
        print(f"[MODELS] Trained models: {len(model_files)}")
        for model_file in model_files:
            print(f"   - {model_file.parent.name}")

    if reports_dir.exists():
        report_files = list(reports_dir.glob("*/evaluation_report.txt"))
        print(f"\n[REPORTS] Evaluation reports: {len(report_files)}")
        for report_file in report_files:
            print(f"   - {report_file.parent.name}")

    # Summary
    comparison_report = reports_dir / "comparison_report.txt"
    if comparison_report.exists():
        print("\n" + "=" * 60)
        print("Pipeline Summary:")
        print("=" * 60)
        with open(comparison_report) as f:
            print(f.read())

        send_notification("Pipeline execution completed successfully!", "success")
    else:
        send_notification("Pipeline execution incomplete", "warning")


@click.command()
@click.option("--dir", default=".", help="Pipeline directory")
def main(dir: str) -> None:
    """Monitor pipeline execution."""
    monitor_pipeline(dir)


if __name__ == "__main__":
    main()

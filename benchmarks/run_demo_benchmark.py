#!/usr/bin/env python3
"""
Simplified benchmark runner for EEG-to-Intent toolkit.

This script demonstrates the benchmark framework with mock implementations
to show the structure and reproducible logging.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.config import (
    BenchmarkConfig,
    BenchmarkResults,
    create_experiment_name,
    set_seed,
    setup_logging,
)


def create_demo_config(
    dataset: str = "bcic_iv_2a",
    model: str = "eegnet",
    protocol: str = "within_subject",
    seed: int = 42,
) -> BenchmarkConfig:
    """Create a demo benchmark configuration."""

    experiment_name = create_experiment_name(
        dataset=dataset,
        model=model,
        protocol=protocol,
        seed=seed,
    )

    return BenchmarkConfig(
        experiment_name=experiment_name,
        dataset_name=dataset,
        model_name=model,
        seed=seed,
        subject_ids=list(range(1, 6)),  # 5 subjects for demo
        evaluation_protocol=protocol,
        max_epochs=10,  # Reduced for demo
        batch_size=32,
        learning_rate=1e-3,
        output_dir=f"outputs/{experiment_name}",
        wandb_project="eeg-to-intent-demo",
        wandb_tags=[dataset, model, protocol],
    )


def run_mock_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Run a mock benchmark to demonstrate the framework."""

    # Set up reproducibility
    set_seed(config.seed, config.deterministic)

    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Starting mock benchmark: {config.experiment_name}")

    # Initialize results
    results = BenchmarkResults(config)

    # Simulate training and evaluation for each subject
    np.random.seed(config.seed)

    for subject_id in config.subject_ids:
        logger.info(f"Processing subject {subject_id}")

        # Mock training process
        logger.info(f"Training {config.model_name} on {config.dataset_name}")

        # Simulate some training epochs
        for epoch in range(min(5, config.max_epochs)):
            train_loss = 1.0 - (epoch * 0.1) + np.random.normal(0, 0.05)
            val_acc = 0.25 + (epoch * 0.1) + np.random.normal(0, 0.02)
            logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, " f"val_acc={val_acc:.3f}")

        # Mock test evaluation
        # Simulate realistic but random performance
        base_accuracy = 0.7  # Base accuracy
        subject_effect = np.random.normal(0, 0.1)  # Subject variability
        noise = np.random.normal(0, 0.05)  # Random noise

        test_accuracy = np.clip(base_accuracy + subject_effect + noise, 0.0, 1.0)
        test_kappa = test_accuracy * 0.8  # Approximate kappa from accuracy
        test_f1 = test_accuracy * 0.9  # Approximate F1 from accuracy

        # Store metrics
        results.add_metric("accuracy", test_accuracy, subject_id)
        results.add_metric("kappa", test_kappa, subject_id)
        results.add_metric("f1_score", test_f1, subject_id)

        # Mock predictions (4 classes for motor imagery)
        n_test_samples = 200
        mock_predictions = np.random.randint(0, 4, n_test_samples)
        mock_targets = np.random.randint(0, 4, n_test_samples)

        # Adjust predictions to match desired accuracy
        n_correct = int(test_accuracy * n_test_samples)
        mock_predictions[:n_correct] = mock_targets[:n_correct]

        results.add_predictions(mock_predictions, mock_targets, subject_id)

        logger.info(f"Subject {subject_id} - Test Accuracy: {test_accuracy:.3f}")

    # Compute overall statistics
    accuracy_values = list(results.metrics["accuracy"].values())
    overall_accuracy = float(np.mean(accuracy_values))
    accuracy_std = float(np.std(accuracy_values))

    results.add_metric("overall_accuracy", overall_accuracy)
    results.add_metric("accuracy_std", accuracy_std)

    logger.info(f"Overall accuracy: {overall_accuracy:.3f} ± {accuracy_std:.3f}")

    # Save results
    results_path = Path(config.output_dir) / "results"
    results.save(results_path)

    logger.info(f"Mock benchmark completed: {config.experiment_name}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run EEG benchmark demo")

    parser.add_argument("--dataset", default="bcic_iv_2a", help="Dataset name")
    parser.add_argument("--model", default="eegnet", help="Model name")
    parser.add_argument(
        "--protocol",
        choices=["within_subject", "cross_subject", "loso"],
        default="within_subject",
        help="Evaluation protocol",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", help="Output directory")

    args = parser.parse_args()

    # Create configuration
    config = create_demo_config(
        dataset=args.dataset,
        model=args.model,
        protocol=args.protocol,
        seed=args.seed,
    )

    if args.output_dir:
        config.output_dir = args.output_dir

    # Run benchmark
    results = run_mock_benchmark(config)

    # Print summary
    summary = results.get_summary()
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Dataset: {summary['dataset']}")
    print(f"Model: {summary['model']}")
    print(f"Protocol: {summary['protocol']}")
    print(f"Seed: {summary['seed']}")
    print(f"Subjects: {len(config.subject_ids or [])}")
    print("-" * 60)

    if "accuracy_mean" in summary:
        print(f"Accuracy: {summary['accuracy_mean']:.3f} ± " f"{summary['accuracy_std']:.3f}")
    if "kappa_mean" in summary:
        print(f"Kappa: {summary['kappa_mean']:.3f} ± " f"{summary['kappa_std']:.3f}")
    if "f1_score_mean" in summary:
        print(f"F1 Score: {summary['f1_score_mean']:.3f} ± " f"{summary['f1_score_std']:.3f}")

    print("-" * 60)
    print(f"Results saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    main()

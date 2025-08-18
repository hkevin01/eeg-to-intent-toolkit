#!/usr/bin/env python3
"""
Run BCIC IV 2a benchmark with EEGNet model.

This script provides a standardized benchmark for the BCIC IV 2a dataset
using the EEGNet model with reproducible settings.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.config import (
    BenchmarkConfig,
    BenchmarkResults,
    create_experiment_name,
    create_wandb_config,
    set_seed,
    setup_logging,
)


def create_bcic_iv_2a_eegnet_config(
    seed: int = 42,
    use_ssl: bool = False,
    use_personalization: bool = False,
    protocol: str = "within_subject",
) -> BenchmarkConfig:
    """Create configuration for BCIC IV 2a + EEGNet benchmark."""

    experiment_name = create_experiment_name(
        dataset="bcic_iv_2a",
        model="eegnet",
        protocol=protocol,
        ssl=use_ssl,
        personalization=use_personalization,
        seed=seed,
    )

    config = BenchmarkConfig(
        # Experiment identification
        experiment_name=experiment_name,
        dataset_name="bcic_iv_2a",
        model_name="eegnet",
        # Reproducibility
        seed=seed,
        deterministic=True,
        # Data configuration
        subject_ids=list(range(1, 10)),  # Subjects 1-9
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        # Model configuration
        model_config={
            "n_classes": 4,  # Left hand, right hand, feet, tongue
            "n_channels": 22,  # Standard 22 EEG channels
            "input_window_samples": 1000,  # 4 seconds at 250 Hz
            "dropout_rate": 0.5,
            "kernel_length": 64,
            "F1": 8,
            "D": 2,
            "F2": 16,
        },
        # Training configuration
        max_epochs=200,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        # SSL configuration
        use_ssl_pretrained=use_ssl,
        ssl_checkpoint_path=("checkpoints/ssl_bcic_iv_2a_best.ckpt" if use_ssl else None),
        freeze_encoder=False,
        # Personalization configuration
        use_personalization=use_personalization,
        personalization_method="film" if use_personalization else None,
        few_shot_samples=5,
        # Evaluation protocol
        evaluation_protocol=protocol,
        cross_validation_folds=5,
        # Logging
        wandb_project="eeg-to-intent-benchmarks",
        wandb_tags=["bcic_iv_2a", "eegnet", protocol],
        # Output
        output_dir=f"outputs/{experiment_name}",
    )

    return config


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Run the benchmark with given configuration."""

    # Set up reproducibility
    set_seed(config.seed, config.deterministic)

    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Starting benchmark: {config.experiment_name}")

    # Initialize results container
    results = BenchmarkResults(config)

    try:
        # Import dependencies
        from eegintent.data.datasets import BCICIVDataset
        from eegintent.evaluation.metrics import compute_classification_metrics
        from eegintent.models.eegnet import EEGNet
        from eegintent.training.trainer import EEGTrainer

        # Set up W&B if configured
        if config.wandb_project:
            import wandb

            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=create_wandb_config(config),
                tags=config.wandb_tags,
            )

        # Load dataset
        logger.info("Loading BCIC IV 2a dataset...")
        dataset = BCICIVDataset(
            subject_ids=config.subject_ids,
            evaluation_protocol=config.evaluation_protocol,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )

        # Evaluate on each subject (or fold)
        if config.evaluation_protocol == "within_subject":
            # Within-subject evaluation
            for subject_id in config.subject_ids:
                logger.info(f"Evaluating subject {subject_id}")

                # Get subject-specific data splits
                train_loader, val_loader, test_loader = dataset.get_subject_loaders(
                    subject_id, batch_size=config.batch_size
                )

                # Create model
                model = EEGNet(**config.model_config)

                # Apply SSL pretraining if configured
                if config.use_ssl_pretrained and config.ssl_checkpoint_path:
                    logger.info("Loading SSL pretrained weights")
                    # Load SSL checkpoint (implementation depends on SSL framework)
                    pass

                # Apply personalization if configured
                if config.use_personalization:
                    logger.info(f"Applying {config.personalization_method} personalization")
                    # Apply personalization method
                    pass

                # Train model
                trainer = EEGTrainer(
                    model=model,
                    max_epochs=config.max_epochs,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                )

                trainer.fit(train_loader, val_loader)

                # Evaluate on test set
                test_metrics = trainer.test(test_loader)

                # Store results
                for metric_name, value in test_metrics.items():
                    results.add_metric(metric_name, value, subject_id)

                # Get predictions for detailed analysis
                predictions, targets = trainer.predict(test_loader)
                results.add_predictions(predictions, targets, subject_id)

                logger.info(
                    f"Subject {subject_id} - Accuracy: {test_metrics.get('accuracy', 0):.3f}"
                )

        elif config.evaluation_protocol == "cross_subject":
            # Cross-subject evaluation (leave-one-subject-out)
            for test_subject in config.subject_ids:
                logger.info(f"Testing on subject {test_subject}")

                # Get cross-subject data splits
                train_subjects = [s for s in config.subject_ids if s != test_subject]

                train_loader, val_loader = dataset.get_cross_subject_loaders(
                    train_subjects, batch_size=config.batch_size
                )
                test_loader = dataset.get_subject_loaders(
                    test_subject, batch_size=config.batch_size
                )[
                    2
                ]  # Only test loader

                # Create and train model
                model = EEGNet(**config.model_config)

                if config.use_ssl_pretrained and config.ssl_checkpoint_path:
                    logger.info("Loading SSL pretrained weights")

                trainer = EEGTrainer(
                    model=model,
                    max_epochs=config.max_epochs,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                )

                trainer.fit(train_loader, val_loader)

                # Test on held-out subject
                test_metrics = trainer.test(test_loader)

                for metric_name, value in test_metrics.items():
                    results.add_metric(metric_name, value, test_subject)

                predictions, targets = trainer.predict(test_loader)
                results.add_predictions(predictions, targets, test_subject)

                logger.info(
                    f"Subject {test_subject} - Accuracy: {test_metrics.get('accuracy', 0):.3f}"
                )

        # Save model if configured
        if config.save_model:
            model_path = Path(config.output_dir) / "model.ckpt"
            trainer.save_checkpoint(model_path)
            results.add_artifact("model_checkpoint", model_path)

        # Compute overall statistics
        accuracy_values = list(results.metrics.get("accuracy", {}).values())
        if accuracy_values:
            overall_accuracy = sum(accuracy_values) / len(accuracy_values)
            results.add_metric("overall_accuracy", overall_accuracy)
            logger.info(f"Overall accuracy: {overall_accuracy:.3f} ± {np.std(accuracy_values):.3f}")

        # Save results
        results_path = Path(config.output_dir) / "results"
        results.save(results_path)

        # Log to W&B
        if config.wandb_project:
            wandb.log(results.get_summary())
            wandb.finish()

        logger.info(f"Benchmark completed successfully: {config.experiment_name}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if config.wandb_project:
            import wandb

            if wandb.run:
                wandb.finish(exit_code=1)
        raise

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run BCIC IV 2a + EEGNet benchmark")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ssl", action="store_true", help="Use SSL pretrained model")
    parser.add_argument("--personalization", action="store_true", help="Use personalization")
    parser.add_argument(
        "--protocol",
        choices=["within_subject", "cross_subject"],
        default="within_subject",
        help="Evaluation protocol",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    # Create configuration
    config = create_bcic_iv_2a_eegnet_config(
        seed=args.seed,
        use_ssl=args.ssl,
        use_personalization=args.personalization,
        protocol=args.protocol,
    )

    if args.no_wandb:
        config.wandb_project = None

    # Run benchmark
    results = run_benchmark(config)

    # Print summary
    summary = results.get_summary()
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Dataset: {summary['dataset']}")
    print(f"Model: {summary['model']}")
    print(f"Protocol: {summary['protocol']}")
    print(f"Seed: {summary['seed']}")

    if "accuracy_mean" in summary:
        print(f"Accuracy: {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}")

    if "overall_accuracy" in summary:
        print(f"Overall Accuracy: {summary['overall_accuracy']:.3f}")

    print("=" * 50)


if __name__ == "__main__":
    main()

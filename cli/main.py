"""Main CLI entry point for tome-raider."""

import click
from rich.console import Console
from rich.table import Table
from loguru import logger
import sys

from tome_raider.core.config import ConfigManager
from tome_raider.core.logging_config import setup_logging
from tome_raider.core.pipeline import DatasetPipeline
from tome_raider.storage.dataset_store import DatasetStore

console = Console()


@click.group()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--profile", "-p", default="default", help="Configuration profile")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, config, profile, verbose):
    """
    GRPO Dataset Builder - Build high-quality training datasets.

    A comprehensive tool for collecting, generating, validating,
    and transforming datasets for GRPO LoRA fine-tuning.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration
    try:
        ctx.obj["config"] = ConfigManager(config, profile)

        # Setup logging
        log_config = ctx.obj["config"].get("logging", {})
        if verbose:
            log_config["level"] = "DEBUG"
        setup_logging(log_config)

        logger.info(f"Initialized with profile: {profile}")
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("config_file")
@click.pass_context
def run(ctx, config_file):
    """Run a pipeline from config file."""
    console.print(f"[cyan]Running pipeline from: {config_file}[/cyan]")

    try:
        pipeline = DatasetPipeline(config_file)
        pipeline.run()

        # Show statistics
        stats = pipeline.get_statistics()
        console.print(f"\n[green]Pipeline complete![/green]")
        console.print(f"Total samples: {stats['sample_count']}")
        console.print(f"Operations executed: {stats['operations_executed']}")

    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        logger.exception("Pipeline execution failed")
        sys.exit(1)


@cli.group()
def dataset():
    """Manage datasets."""
    pass


@dataset.command("list")
@click.pass_context
def list_datasets(ctx):
    """List all datasets."""
    config = ctx.obj["config"]
    store = DatasetStore(config.get("storage.base_path", "./datasets"))

    datasets = store.list_datasets()

    if not datasets:
        console.print("[yellow]No datasets found[/yellow]")
        return

    # Create table
    table = Table(title="Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Samples", style="green")
    table.add_column("Format", style="yellow")
    table.add_column("Created", style="blue")

    for ds in datasets:
        table.add_row(
            ds["name"],
            str(ds["sample_count"]),
            ds["format"],
            ds["created_at"][:10]
        )

    console.print(table)


@dataset.command("info")
@click.argument("name")
@click.pass_context
def dataset_info(ctx, name):
    """Show dataset information."""
    config = ctx.obj["config"]
    store = DatasetStore(config.get("storage.base_path", "./datasets"))

    try:
        dataset = store.load(name)
        info = store.index.get(name, {})

        console.print(f"\n[cyan]Dataset: {name}[/cyan]")
        console.print(f"Samples: {len(dataset)}")
        console.print(f"Format: {info.get('format', 'unknown')}")
        console.print(f"Created: {info.get('created_at', 'unknown')}")

        # Show metadata
        metadata = info.get("metadata", {})
        if metadata:
            console.print("\n[yellow]Metadata:[/yellow]")
            for key, value in metadata.items():
                console.print(f"  {key}: {value}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@dataset.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure?")
@click.pass_context
def delete_dataset(ctx, name):
    """Delete a dataset."""
    config = ctx.obj["config"]
    store = DatasetStore(config.get("storage.base_path", "./datasets"))

    try:
        store.delete(name)
        console.print(f"[green]Deleted dataset: {name}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.group()
def generate():
    """Generate synthetic data."""
    pass


@generate.command("self-instruct")
@click.option("--model", "-m", required=True, help="Path to model file")
@click.option("--count", "-n", default=100, help="Number of samples to generate")
@click.option("--output", "-o", help="Output dataset name")
@click.pass_context
def generate_self_instruct(ctx, model, count, output):
    """Generate using self-instruct strategy."""
    from tome_raider.generation.generator import DataGenerator

    console.print(f"[cyan]Generating {count} samples with self-instruct...[/cyan]")

    try:
        config = ctx.obj["config"]
        generator = DataGenerator(config.to_dict())

        samples = generator.generate(
            strategy="self_instruct",
            model=model,
            target_count=count
        )

        console.print(f"[green]Generated {len(samples)} samples[/green]")

        # Save if output specified
        if output:
            store = DatasetStore(config.get("storage.base_path", "./datasets"))
            store.save(samples, output)
            console.print(f"[green]Saved to dataset: {output}[/green]")

        generator.close()

    except Exception as e:
        console.print(f"[red]Generation failed: {e}[/red]")
        logger.exception("Generation failed")
        sys.exit(1)


@cli.group()
def validate():
    """Validate datasets."""
    pass


@validate.command("check")
@click.argument("dataset_name")
@click.option("--strict", is_flag=True, help="Strict validation mode")
@click.pass_context
def validate_check(ctx, dataset_name, strict):
    """Validate a dataset."""
    from tome_raider.quality.validator import DatasetValidator

    console.print(f"[cyan]Validating dataset: {dataset_name}[/cyan]")

    try:
        config = ctx.obj["config"]
        store = DatasetStore(config.get("storage.base_path", "./datasets"))
        dataset = store.load(dataset_name)

        validator = DatasetValidator(config.get("validation", {}))
        result = validator.validate_all(dataset)

        # Show results
        console.print(f"\nTotal samples: {result['total']}")
        console.print(f"[green]Valid: {result['valid']}[/green]")
        console.print(f"[red]Invalid: {result['invalid']}[/red]")

        if result['errors']:
            console.print(f"\n[red]Errors ({len(result['errors'])}):[/red]")
            for error in result['errors'][:10]:  # Show first 10
                console.print(f"  - {error}")

            if len(result['errors']) > 10:
                console.print(f"  ... and {len(result['errors']) - 10} more")

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        sys.exit(1)


@validate.command("repair")
@click.argument("dataset_name")
@click.option("--strategy", "-s", default="truncate",
              type=click.Choice(["truncate", "summarize", "split"]),
              help="Repair strategy")
@click.option("--model", "-m", help="Path to model file (required for summarize strategy)")
@click.option("--output", "-o", help="Output dataset name (default: overwrites original)")
@click.option("--backup/--no-backup", default=True, help="Create backup before repairing")
@click.option("--dry-run", is_flag=True, help="Preview repairs without saving")
@click.pass_context
def validate_repair(ctx, dataset_name, strategy, model, output, backup, dry_run):
    """Repair a dataset with validation errors."""
    from tome_raider.quality.repairer import DatasetRepairer
    from tome_raider.quality.validator import DatasetValidator

    console.print(f"[cyan]Repairing dataset: {dataset_name}[/cyan]")
    console.print(f"Strategy: {strategy}")

    # Validate model requirement for summarize strategy
    if strategy == "summarize" and not model:
        console.print("[red]Error: --model is required for summarize strategy[/red]")
        console.print("Example: tome-raider validate repair dataset_name --strategy summarize --model path/to/model.gguf")
        sys.exit(1)

    try:
        config = ctx.obj["config"]
        store = DatasetStore(config.get("storage.base_path", "./datasets"))
        dataset = store.load(dataset_name)

        # Create repairer
        validator = DatasetValidator(config.get("validation", {}))

        # Add model path to config for summarize strategy
        repair_config = config.to_dict()
        if strategy == "summarize":
            if "repair" not in repair_config:
                repair_config["repair"] = {}
            repair_config["repair"]["model_path"] = model
            console.print(f"Using model: {model}")

        repairer = DatasetRepairer(
            validator=validator,
            strategy=strategy,
            config=repair_config
        )

        # Repair and validate
        console.print("\n[yellow]Analyzing dataset...[/yellow]")
        result = repairer.repair_and_validate(dataset)

        # Show before/after statistics
        before = result["before_validation"]
        after = result["after_validation"]
        repair_stats = result["repair_statistics"]

        console.print("\n[cyan]Before Repair:[/cyan]")
        console.print(f"  Valid: {before['valid']}/{before['total']}")
        console.print(f"  Invalid: {before['invalid']}/{before['total']}")

        console.print("\n[cyan]After Repair:[/cyan]")
        console.print(f"  Valid: {after['valid']}/{after['total']}")
        console.print(f"  Invalid: {after['invalid']}/{after['total']}")

        console.print(f"\n[green]Improvement:[/green]")
        console.print(f"  Fixed: {result['improvement']['invalid_count']} samples")
        console.print(f"  Already valid: {repair_stats['already_valid']}")
        console.print(f"  Repaired: {repair_stats['repaired']}")
        console.print(f"  Failed to repair: {repair_stats['failed']}")

        # Show sample changes
        if repair_stats['changes']:
            console.print(f"\n[yellow]Sample changes (first 5):[/yellow]")
            for change_info in repair_stats['changes'][:5]:
                idx = change_info['sample_index']
                changes = change_info['changes']
                console.print(f"  Sample {idx}:")
                for change in changes:
                    console.print(f"    - {change}")

            if len(repair_stats['changes']) > 5:
                remaining = len(repair_stats['changes']) - 5
                console.print(f"  ... and {remaining} more")

        # Show remaining errors
        if after['errors']:
            console.print(f"\n[red]Remaining errors ({len(after['errors'])}):[/red]")
            for error in after['errors'][:5]:
                console.print(f"  - {error}")
            if len(after['errors']) > 5:
                console.print(f"  ... and {len(after['errors']) - 5} more")

        # Save if not dry-run
        if not dry_run:
            # Create backup if requested
            if backup and not output:
                backup_name = f"{dataset_name}_backup"
                store.save(dataset, backup_name)
                console.print(f"\n[yellow]Backup saved as: {backup_name}[/yellow]")

            # Save repaired dataset
            output_name = output or dataset_name
            store.save(result["dataset"], output_name)
            console.print(f"\n[green]Repaired dataset saved as: {output_name}[/green]")
        else:
            console.print("\n[yellow]Dry-run mode: No changes saved[/yellow]")

        # Cleanup resources (e.g., unload models)
        repairer.cleanup()

    except Exception as e:
        console.print(f"[red]Repair failed: {e}[/red]")
        logger.exception("Repair failed")
        sys.exit(1)


@cli.group()
def quality():
    """Quality scoring and analysis."""
    pass


@quality.command("score")
@click.argument("dataset_name")
@click.option("--save", is_flag=True, help="Save scores to dataset")
@click.pass_context
def quality_score(ctx, dataset_name, save):
    """Score dataset quality."""
    from tome_raider.quality.quality_scorer import QualityScorer

    console.print(f"[cyan]Scoring quality for: {dataset_name}[/cyan]")

    try:
        config = ctx.obj["config"]
        store = DatasetStore(config.get("storage.base_path", "./datasets"))
        dataset = store.load(dataset_name)

        scorer = QualityScorer()

        # Score all samples
        with console.status("[bold green]Scoring samples...") as status:
            for idx, sample in enumerate(dataset):
                score = scorer.score_sample(sample)
                sample.metadata.quality_score = score.overall

                if (idx + 1) % 10 == 0:
                    status.update(f"[bold green]Scoring... {idx + 1}/{len(dataset)}")

        # Show statistics
        scores = [s.metadata.quality_score for s in dataset if s.metadata.quality_score]

        if scores:
            import numpy as np
            console.print(f"\n[green]Scoring complete![/green]")
            console.print(f"Mean: {np.mean(scores):.3f}")
            console.print(f"Std: {np.std(scores):.3f}")
            console.print(f"Min: {np.min(scores):.3f}")
            console.print(f"Max: {np.max(scores):.3f}")

        # Save if requested
        if save:
            store.update(dataset_name, dataset)
            console.print(f"[green]Scores saved to dataset[/green]")

    except Exception as e:
        console.print(f"[red]Scoring failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("dataset_name")
@click.option("--readonly", is_flag=True, help="Open in read-only mode")
@click.option("--validate", is_flag=True, help="Run validation automatically")
@click.option("--filter-status", type=click.Choice(["pending", "approved", "rejected"]),
              help="Filter by review status")
@click.pass_context
def review(ctx, dataset_name, readonly, validate, filter_status):
    """Launch interactive review interface for a dataset."""
    try:
        config = ctx.obj["config"]
        store = DatasetStore(config.get("storage.base_path", "./datasets"))

        console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")
        dataset = store.load(dataset_name)

        console.print(f"[green]Loaded {len(dataset)} samples[/green]")

        if readonly:
            console.print("[yellow]Read-only mode enabled[/yellow]")

        # Launch TUI
        from tome_raider.review.interactive import launch_review_ui

        launch_review_ui(
            dataset_name=dataset_name,
            dataset=dataset,
            store=store,
            readonly=readonly,
            auto_validate=validate
        )

    except Exception as e:
        console.print(f"[red]Review failed: {e}[/red]")
        logger.exception("Review failed")
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    console.print("[cyan]GRPO Dataset Builder v0.1.0[/cyan]")
    console.print("A comprehensive tool for building training datasets")


if __name__ == "__main__":
    cli(obj={})

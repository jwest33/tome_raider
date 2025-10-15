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
def version():
    """Show version information."""
    console.print("[cyan]GRPO Dataset Builder v0.1.0[/cyan]")
    console.print("A comprehensive tool for building training datasets")


if __name__ == "__main__":
    cli(obj={})

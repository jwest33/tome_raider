# Tome Raider - GRPO Dataset Builder

An LLM powered dataset builder for creating high-quality training datasets for GRPO LoRA fine-tuning. Build professional-grade datasets with data collection, generation, quality control, and transformation capabilities.

## Features

- **Multiple Data Sources**: Load from files (JSON, JSONL, CSV, Parquet, TXT, PDF), HuggingFace datasets, or web scraping (Stack Overflow, GitHub)
- **6 Generation Strategies**: Self-Instruct, Evol-Instruct, Distillation, Response Generation, Instruction Generation, Chain-of-Thought
- **Quality Control**: Strict validation, multi-metric quality scoring, exact and fuzzy deduplication
- **Flexible Storage**: File-based storage with fast indexing and filtering
- **Config-Driven Pipelines**: YAML-based pipeline configuration for reproducible workflows
- **Comprehensive Metadata**: Track source, quality scores, tags, and review status
- **CLI Interface**: Rich terminal interface for all operations
- **Extensible**: Modular architecture for easy extension

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd data_gen

# Install dependencies
pip install -r requirements.txt

# Install as package
pip install -e .
```

## Quick Start

### 1. Generate Data from Scratch

```bash
# Generate 100 samples using self-instruct
dataset-builder generate self-instruct --model c:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-Q8_0.gguf --count 100 --output test_001
```

### 2. Run a Pipeline

```bash
# Execute a complete pipeline from config
dataset-builder run examples/math_dataset_pipeline.yaml
```

### 3. Validate and Score Quality

```bash
# Validate dataset
dataset-builder validate check my_dataset --strict

# Score quality
dataset-builder quality score my_dataset --save
```

### 4. Manage Datasets

```bash
# List all datasets
dataset-builder dataset list

# Show dataset info
dataset-builder dataset info my_dataset

# Delete dataset
dataset-builder dataset delete my_dataset
```

## Usage Guide

### Data Sources

#### Load from Files

```python
from dataset_builder.sources.file_loader import FileLoader

# Load from JSON/JSONL/CSV/Parquet
loader = FileLoader({"path": "data/*.jsonl"})
samples = list(loader.load())
```

#### Load from HuggingFace

```python
from dataset_builder.sources.dataset_loader import DatasetLoader

# Load HuggingFace dataset
loader = DatasetLoader({
    "dataset": "openai/gsm8k",
    "split": "train"
})
samples = list(loader.load())
```

#### Web Scraping

```python
from dataset_builder.sources.scrapers.stackoverflow import StackOverflowScraper

# Scrape Stack Overflow
scraper = StackOverflowScraper({
    "tags": ["python", "machine-learning"],
    "max_samples": 100
})
samples = list(scraper.load())
```

### Data Generation

#### Self-Instruct

```python
from dataset_builder.generation.generator import DataGenerator

generator = DataGenerator({})
samples = generator.generate(
    strategy="self_instruct",
    model="models/mistral-7b-instruct.gguf",
    target_count=100
)
```

#### Evol-Instruct

```python
samples = generator.generate(
    strategy="evol_instruct",
    model="models/mistral-7b-instruct.gguf",
    base_instructions="instructions.jsonl",
    evolution_rounds=3
)
```

#### Chain-of-Thought

```python
samples = generator.generate(
    strategy="chain_of_thought",
    model="models/deepseek-coder-33b.gguf",
    problems="problems.jsonl"
)
```

### Quality Control

#### Validation

```python
from dataset_builder.quality.validator import DatasetValidator

validator = DatasetValidator({"strict_mode": True})
result = validator.validate_all(samples)

print(f"Valid: {result['valid']}/{result['total']}")
```

#### Quality Scoring

```python
from dataset_builder.quality.quality_scorer import QualityScorer

scorer = QualityScorer()

for sample in samples:
    score = scorer.score_sample(sample)
    sample.metadata.quality_score = score.overall
```

#### Deduplication

```python
from dataset_builder.quality.deduplicator import Deduplicator

dedup = Deduplicator()

# Remove exact duplicates
unique, removed = dedup.remove_exact_duplicates(samples)

# Remove near-duplicates
unique, removed = dedup.remove_near_duplicates(samples, threshold=0.85)
```

### Storage and Retrieval

```python
from dataset_builder.storage.dataset_store import DatasetStore

store = DatasetStore()

# Save dataset
store.save(samples, "my_dataset", format="jsonl")

# Load dataset
loaded_samples = store.load("my_dataset")

# List all datasets
datasets = store.list_datasets()
```

### Config-Driven Pipelines

Create a pipeline configuration file:

```yaml
# my_pipeline.yaml
name: "My Dataset Pipeline"

operations:
  - name: "Load data"
    type: "source"
    config:
      source_type: "file"
      path: "data/*.jsonl"

  - name: "Generate more samples"
    type: "generate"
    config:
      strategy: "self_instruct"
      model: "models/mistral-7b.gguf"
      target_count: 500

  - name: "Validate"
    type: "validate"
    config:
      strict: true
      remove_invalid: true

  - name: "Score quality"
    type: "quality"
    config: {}

  - name: "Deduplicate"
    type: "deduplicate"
    config:
      exact: true
      near: true

  - name: "Filter high quality"
    type: "filter"
    config:
      quality_min: 0.7

  - name: "Save"
    type: "save"
    config:
      name: "final_dataset"
      format: "jsonl"
```

Run the pipeline:

```bash
dataset-builder run my_pipeline.yaml
```

## CLI Commands

### Dataset Management
- `dataset list` - List all datasets
- `dataset info <name>` - Show dataset information
- `dataset delete <name>` - Delete a dataset

### Generation
- `generate self-instruct` - Generate using self-instruct
- More strategies available via pipelines

### Validation
- `validate check <dataset>` - Validate dataset

### Quality
- `quality score <dataset>` - Score dataset quality

### Pipeline
- `run <config>` - Run pipeline from config file

## Generation Strategies

1. **Self-Instruct**: Generate new instructions from seed tasks
2. **Evol-Instruct**: Evolve instructions through multiple rounds
3. **Distillation**: Use teacher model to generate training data
4. **Response Generation**: Generate responses for existing instructions
5. **Instruction Generation**: Extract instructions from documents
6. **Chain-of-Thought**: Generate step-by-step reasoning

## Configuration

### Profiles

- `default` - Balanced settings
- `dev` - Development mode (verbose logging, lenient validation)
- `prod` - Production mode (strict validation, larger batches)

Use profiles:

```bash
dataset-builder --profile dev generate ...
```

### Environment Variables

Override config with environment variables:

```bash
export DB_STORAGE_BASE_PATH="./my_datasets"
export DB_LOGGING_LEVEL="DEBUG"
```

## Quality Metrics

The quality scorer evaluates samples across multiple dimensions:

- **Instruction Clarity**: Grammar, readability, specificity
- **Instruction Complexity**: Vocabulary richness, concept depth
- **Response Completeness**: Addresses all aspects
- **Response Coherence**: Logical flow and structure
- **Alignment**: Response matches instruction intent
- **Diversity**: Vocabulary variety

## Contributing

Contributions are welcome! The codebase is designed to be extensible:

- Add new data sources by extending `DataSource`
- Add new generation strategies by extending `GenerationStrategy`
- Add new validators by extending the validation system
- Add new transformations by creating transformation modules

## Example Workflows

### Build a Math Dataset

```bash
# 1. Load seed data
# 2. Evolve instructions
# 3. Validate and deduplicate
# 4. Score quality
# 5. Filter and save

dataset-builder run examples/math_dataset_pipeline.yaml
```

### Scrape and Clean Web Data

```python
from dataset_builder.sources.scrapers.stackoverflow import StackOverflowScraper
from dataset_builder.quality.deduplicator import Deduplicator
from dataset_builder.storage.dataset_store import DatasetStore

# Scrape
scraper = StackOverflowScraper({"tags": ["python"], "max_samples": 1000})
samples = list(scraper.load())

# Deduplicate
dedup = Deduplicator()
samples, _ = dedup.remove_exact_duplicates(samples)

# Save
store = DatasetStore()
store.save(samples, "stackoverflow_python")
```

### Generate with Distillation

```python
from dataset_builder.generation.generator import DataGenerator

generator = DataGenerator({})

# Use large teacher model
samples = generator.generate(
    strategy="distillation",
    model="models/mixtral-8x7b.gguf",
    prompts="prompts.jsonl",
    batch_size=8
)
```

## Documentation

- `DESIGN.md` - Comprehensive design document
- `IMPLEMENTATION_STATUS.md` - Implementation progress and roadmap
- `examples/` - Example configurations and workflows

## Troubleshooting

### llama-server not found

Ensure llama-server is installed and in PATH:

```bash
which llama-server
```

### Module import errors

Install in development mode:

```bash
pip install -e .
```

### Permission errors

Check directory permissions for `datasets/`, `logs/`, `.cache/`

## Examples

- [Examples](examples/) - Sample configurations

## License

Created and maintained by **jwest33** ([loracraft.org](https://loracraft.org))  
Licensed under the [MIT License](./LICENSE).  

## Acknowledgments

Built with:
- Click for CLI
- Rich for terminal UI
- Loguru for logging
- llama.cpp for local LLM inference
- HuggingFace datasets for data loading

---

**GRPO Dataset Builder** - Build professional-grade training datasets with ease.

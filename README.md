# BioVerify: Computer Vision Pipeline for Species Identification in Crowdsourced Wildlife Images

A detection-then-classification pipeline for validating species identity in noisy iNaturalist predator-prey interaction images using LLaMA-generated prompts, GroundingDINO localization, and BioCLIP2 taxonomic classification.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![BioCLIP2](https://img.shields.io/badge/BioCLIP2-Taxonomic%20Classification-green)
![GroundingDINO](https://img.shields.io/badge/GroundingDINO-Open--Vocabulary%20Detection-purple)
![LLaMA](https://img.shields.io/badge/LLaMA-Prompt%20Engineering-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Wildlife%20AI-teal)
![Applied ML](https://img.shields.io/badge/Applied%20ML-Data%20Validation-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- [Purpose](#purpose)
- [Pipeline Overview](#pipeline-overview)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Data and Model Availability](#data-and-model-availability)
- [Project Notes and Internal Documentation](#project-notes-and-internal-documentation)
- [Future Scope](#future-scope)
- [Contributors](#contributors)
- [Contributing](#contributing)
- [License](#license)

## Purpose

BioVerify was created to support species-identity validation in the Who-Eats-Whom iNaturalist interaction dataset.

Crowdsourced biodiversity datasets contain valuable ecological interaction information, but the images and metadata can be noisy. Errors may include incorrect species identification, predator/prey direction mistakes, non-predation interactions labeled as predation, unclear images, small organisms, and taxonomic ambiguity. Since downstream food-web or Neo4j graph analysis depends heavily on the correctness of the input data, validation is a necessary preprocessing step.

This repository focuses on **Stage 1: vision-based species identity verification**. The planned Stage 2 is interaction validity verification using RAG / Graph RAG to evaluate whether predator-prey claims are biologically plausible.

## Pipeline Overview

The BioVerify workflow combines dataset cleaning, open-vocabulary object localization, crop extraction, and taxonomic image classification:

1. iNaturalist / Who-Eats-Whom data collection and cleaning
2. Observation and partner-observation mapping
3. LLaMA prompt generation for organism-aware detection prompts
4. GroundingDINO open-vocabulary object localization
5. Bounding-box crop generation
6. BioCLIP2 classification on cropped organism regions
7. Taxonomic matching against the user-provided iNaturalist label
8. Evaluation using species-level accuracy, hierarchical taxonomic accuracy, taxonomic distance, and confidence-score comparison

```text
iNaturalist / Who-Eats-Whom Records
        |
        v
Data Cleaning + Observation Mapping
        |
        v
LLaMA Prompt Generation
        |
        v
GroundingDINO Bounding Box Detection
        |
        v
Crop Extraction
        |
        v
BioCLIP2 Taxonomic Classification
        |
        v
Species / Genus / Family-Level Matching
        |
        v
Pass, Near-Match, or Human Review Flag
```

## Key Results

### Baseline BioCLIP2 on Full Images

- Total images evaluated: 24,807
- Correct species-level predictions: 13,638
- Species-level accuracy: 54.98%
- Error rate: 45.02%

### BioVerify Crop-Based Pipeline

- Total images evaluated: 24,803
- Correct species-level predictions: 17,840
- Species-level accuracy: 71.93%
- Error rate: 28.07%
- Improvement over full-image baseline: approximately +16.95 percentage points

### Hierarchical Accuracy for BioVerify Full Dataset

| Taxonomic Level | Accuracy |
| --------------- | -------: |
| Kingdom         |   98.18% |
| Phylum          |   97.16% |
| Class           |   86.49% |
| Family          |   86.36% |
| Order           |   83.60% |
| Genus           |   81.31% |
| Species         |   71.93% |

### GroundingDINO Prompt Engineering Improvement - Test Case Analysis

- Generic prompts: 63% clear detections, 37% ambiguous, 0 missed
- LLaMA-optimized prompts: 84% clear detections, approximately 15% ambiguous
- Better text prompts improved localization quality before BioCLIP2 classification.

### Taxonomic Distance Analysis

- 74,746 crop / bounding-box groups analyzed
- Up to 5 predictions per crop
- 373,730 top-k prediction rows analyzed
- Mean taxonomic distance: 5.36
- Top-1 predictions were taxonomically closer on average than lower-ranked predictions.
- Around 46.87% of top-k predictions were exact species, same-genus, or same-family matches.
- Model errors were often taxonomically structured rather than random.

### Confidence Score Analysis

- Compared full-image BioCLIP2 confidence against crop-based BioVerify confidence only where baseline was already correct.
- Baseline-correct matched records analyzed: 13,852
- BioVerify had higher confidence in 8,449 cases, or 61.0%.
- Baseline had equal or higher confidence in 5,403 cases, or 39.0%.
- Localization often gave BioCLIP2 a stronger organism-focused visual signal.

## Tech Stack

### Languages and Core Libraries

- Python
- pandas
- NumPy
- tqdm
- Jupyter Notebook

### Machine Learning / Deep Learning

- PyTorch
- Hugging Face Transformers
- BioCLIP2
- GroundingDINO
- LLaMA
- OpenCLIP / CLIP-style image-text matching

### Computer Vision

- Open-vocabulary object detection
- Bounding-box crop extraction
- Optional / experimental SAM2 segmentation

### Data and Evaluation

- CSV-based data processing
- Taxonomic hierarchy matching
- Confidence score analysis
- Taxonomic distance analysis
- Matplotlib-based evaluation and visualization notebooks

## Repository Structure

The repository contains notebooks, scripts, CSV metadata, prompt outputs, detection outputs, BioCLIP2 evaluation results, and downstream analysis outputs. Large image datasets, derived crop image folders, and model weights are intentionally excluded.

```text
BioVerify/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- trial.py
|-- EDA/
|   |-- README.txt
|   |-- Cleaning.ipynb
|   |-- dataset_eda.ipynb
|   |-- link.ipynb
|   |-- obs_check.ipynb
|   |-- Phylum_Class_Distribution.ipynb
|   |-- Dataset.csv
|   |-- check.csv
|   |-- correct.csv
|   |-- final_df.csv
|   |-- invalid_urls.csv
|   |-- link1.csv
|   |-- obs_check.csv
|   |-- observations-704882.csv
|   |-- project_41347_research_observations.csv
|   |-- project_41347_research_photos.csv
|   |-- phylum_samples.csv
|   |-- phylum_samples_1.csv
|   |-- phylum_samples - Copy.csv
|   |-- Data/
|   |-- outputs/
|-- Scraping/
|   |-- scrape.ipynb
|   |-- scrape copy.ipynb
|   |-- obs_yes_with_license.csv
|   |-- obs_no_unfiltered.csv
|-- prompts/
|   |-- dataset_with_prompts.csv
|   |-- dataset_with_prompts_1phrase.csv
|   |-- dataset_with_prompts_a100.csv
|   |-- image_extraction.py
|   |-- load_dataset.py
|   |-- missed_detections.csv
|   |-- run_load_dataset.sh
|-- notebooks/
|   |-- classification.ipynb
|   |-- classification_baseline.ipynb
|   |-- classification_crop.ipynb
|   |-- classification_crop2.ipynb
|   |-- crop.ipynb
|   |-- crop2.ipynb
|   |-- Visualization.ipynb
|   |-- bounding.ipynb
|   |-- results_crop_eval.ipynb
|   |-- bio_clip/
|   |-- bounding_boxes/
|   |-- prompt_engineering/
|   |-- verfication/
|-- results/
|   |-- baseline_eval.ipynb
|   |-- results_crop_eval.ipynb
|   |-- results_eval.ipynb
|   |-- results_check.ipynb
|   |-- baseline.csv
|   |-- baseline_metadata.csv
|   |-- eval.csv
|   |-- matched.csv
|   |-- results_baseline.csv
|   |-- results_crop.csv
|   |-- results_crop_updated.csv
|   |-- results_crop_old.csv
|   |-- results_crop_new.csv
|   |-- unmatch_class.csv
|   |-- unmatch_family.csv
|   |-- unmatch_order.csv
|   |-- outputs_bioclip/
|   |-- outputs_bioclip_crop/
|   |-- outputs_gd/
|-- outputs/
|   |-- sam_boxes*.csv
|   |-- sam/
|-- analysis_outputs/
|   |-- aggregate_taxonomic_distance_summary.csv
|   |-- prediction_level_taxonomic_distances.csv
|   |-- top5_pairwise_confusion_by_crop.csv
|   |-- topk_correct_present_distance_summary.csv
|-- images/
```

### Key Files and Folders

| Path                                                                                                                           | Purpose                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `EDA/README.txt`                                                                                                               | Documents the iNaturalist export query and column meanings for the Who-Eats-Whom observations.                                                   |
| `EDA/Cleaning.ipynb`                                                                                                           | Pulls observation data from the iNaturalist API and performs preprocessing / cleaning.                                                           |
| `EDA/link.ipynb`                                                                                                               | Extracts and checks partner-observation links from iNaturalist observation fields.                                                               |
| `EDA/obs_check.ipynb`                                                                                                          | Evaluates whether observation and partner IDs are present in the processed dataset.                                                              |
| `EDA/Phylum_Class_Distribution.ipynb`                                                                                          | Computes phylum/class distribution summaries and sample selections.                                                                              |
| `EDA/Dataset.csv`                                                                                                              | Main processed CSV used by downstream pipeline stages; includes observation IDs, photo URLs, labels, predator/prey fields, and taxonomy columns. |
| `EDA/check.csv`                                                                                                                | Observation/partner-ID verification table with match flags.                                                                                      |
| `EDA/correct.csv`                                                                                                              | Invalid-link correction/check table with `Invalid_link`, `category`, and `correct` columns.                                                      |
| `EDA/link1.csv`                                                                                                                | Observation records with extracted `partner_ids`.                                                                                                |
| `EDA/observations-704882.csv`                                                                                                  | iNaturalist Who-Eats-Whom observation export.                                                                                                    |
| `EDA/project_41347_research_observations.csv`                                                                                  | Research-grade observation metadata pulled from the iNaturalist API.                                                                             |
| `EDA/project_41347_research_photos.csv`                                                                                        | Photo metadata and image URLs for the research-grade observations.                                                                               |
| `EDA/Data/`                                                                                                                    | Additional dataset snapshots, including `Dataset-03-02-26.csv`, `final_df.csv`, and another observation export.                                  |
| `Scraping/`                                                                                                                    | iNaturalist scraping notebooks and large raw API result CSVs, including licensed and unfiltered observation outputs.                             |
| `prompts/dataset_with_prompts.csv`, `prompts/dataset_with_prompts_1phrase.csv`, `prompts/dataset_with_prompts_a100.csv`        | Dataset variants augmented with `grounding_dino_prompt` text for organism-aware localization.                                                    |
| `prompts/load_dataset.py`                                                                                                      | Downloads images from processed dataset URLs using a priority order of original, large, medium, then square image URLs.                          |
| `prompts/image_extraction.py`                                                                                                  | Small image download/extraction helper script.                                                                                                   |
| `notebooks/classification.ipynb`                                                                                               | BioCLIP2 species classification workflow for cropped image inputs.                                                                               |
| `notebooks/classification_baseline.ipynb`                                                                                      | Baseline BioCLIP2 workflow for full-image classification.                                                                                        |
| `notebooks/classification_crop.ipynb`                                                                                          | BioCLIP2 workflow for cropped bounding-box outputs.                                                                                              |
| `notebooks/crop.ipynb`                                                                                                         | Crop-generation notebook that uses bounding-box coordinates to extract organism regions.                                                         |
| `notebooks/Visualization.ipynb`                                                                                                | Visualizes classification and crop results with image grids and annotations.                                                                     |
| `notebooks/prompt_engineering/`                                                                                                | LLaMA / Hugging Face prompt-engineering notebooks and Python scripts for creating GroundingDINO detection prompts.                               |
| `notebooks/bounding_boxes/bounding_boxes_full_dataset.py`                                                                      | Runs GroundingDINO with generated prompts over the image dataset and writes bounding-box and summary CSVs.                                       |
| `results/baseline_eval.ipynb`                                                                                                  | Baseline full-image BioCLIP2 evaluation notebook.                                                                                                |
| `results/baseline.csv`                                                                                                         | Baseline evaluation output with top-k predictions and taxonomic match columns.                                                                   |
| `results/results_crop_eval.ipynb`                                                                                              | Crop-based pipeline evaluation notebook for hierarchical taxonomic matching and accuracy analysis.                                               |
| `results/results_crop.csv`, `results/results_crop_updated.csv`, `results/results_crop_old.csv`, `results/results_crop_new.csv` | Crop-based BioCLIP2 prediction and evaluation outputs.                                                                                           |
| `results/outputs_bioclip/`                                                                                                     | Baseline BioCLIP2 full-image model outputs.                                                                                                      |
| `results/outputs_bioclip_crop/`                                                                                                | Crop-based BioCLIP2 prediction outputs and matched result tables.                                                                                |
| `results/outputs_gd/`                                                                                                          | GroundingDINO bounding-box outputs, including full-dataset box and summary CSVs.                                                                 |
| `results/unmatch_class.csv`, `results/unmatch_family.csv`, `results/unmatch_order.csv`                                         | Taxonomic mismatch tables grouped by unmatched class, family, or order fields.                                                                   |
| `outputs/`                                                                                                                     | Experimental SAM/SAM2-style segmentation and bounding-box outputs.                                                                               |
| `analysis_outputs/`                                                                                                            | Downstream taxonomic-distance and top-k confusion analysis outputs.                                                                              |
| `images/`                                                                                                                      | Small local/sample image folder, not the full iNaturalist image dataset.                                                                         |

### Expected / Not Included / Naming Notes

- `models/` is not present in the repository. Model weights are excluded by `.gitignore`.
- The full image dataset, crop image outputs, SAM image outputs, and model weight files are excluded by `.gitignore`.

## Data and Model Availability

The raw iNaturalist image dataset is not uploaded to GitHub because of iNaturalist licensing constraints and GitHub storage limits. Cropped output images such as `output_crop` / crop-image folders are also not uploaded because they are derived from the original licensed image dataset and can be large.

Some generated output files may be stored as compressed ZIP archives because of file size and GitHub storage restrictions. Extract these archives locally before inspecting or reusing the contained outputs.

Model weights are not uploaded because of storage limits. Users must configure external model access before reproducing the full pipeline.

Users should download data from the official iNaturalist / Who-Eats-Whom sources and the iNaturalist Open Data registry:

```text
https://registry.opendata.aws/inaturalist-open-data/
```

For model setup, refer to the official documentation or model pages for:

- BioCLIP2 / Imageomics
- GroundingDINO
- LLaMA
- Hugging Face Transformers
- PyTorch

This repository should not be treated as fully runnable without external data, local image directories, model weights, and path/configuration updates.

## Project Notes and Internal Documentation

Detailed development notes, experiment logs, and intermediate reasoning are documented in Notion for personal research tracking:

https://www.notion.so/Data-Validation-Pipeline-30336e25953d807f933accac74e7dec6

## Future Scope

- Add RAG-based interaction validation for checking whether predator-prey claims are biologically plausible.
- Build Graph RAG over the verified food-web graph.
- Add an expert-review interface for uncertain or taxonomically distant predictions.
- Improve threshold calibration using confidence score, taxonomic distance, and crop quality.
- Add automated crop-quality scoring.
- Improve handling of fungi, small organisms, parasites, and visually ambiguous taxa.
- Add reproducible config files and CLI scripts.

## Contributors

- **Surabhi Ravindran Nair** 
- **Lavanya Middha**
- Under supervision of **Dr. Aditi Mallavarapu**

## Contributing

Contributions should keep the pipeline reproducible, documented, and clear about data/model requirements.

1. Fork the repository.
2. Create a feature branch.
3. Make changes with clear commit messages.
4. Add documentation for new notebooks or scripts.
5. Open a pull request explaining the change.

```bash
git checkout -b feature/your-feature-name
git add .
git commit -m "Add clear description of change"
git push origin feature/your-feature-name
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.

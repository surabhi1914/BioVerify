# %% [markdown]
# # Taxonomic Confusion Metrics for BioVerify / BioCLIP2 Results
#
# This script calculates taxonomic distance metrics for prediction-level
# BioCLIP2 outputs and top-k crop-level confusion. It is written as a
# notebook-style Python file: in VS Code/Jupyter-compatible editors, each
# `# %%` block can be run as a separate cell.

# %%
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# %% [markdown]
# ## 1. Configuration
#
# The default paths match your project layout. The script never overwrites the
# original input files. All generated CSV files are written to `analysis_outputs/`.

# %%
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # When running interactively in a notebook, __file__ is not defined.
    SCRIPT_DIR = Path.cwd()

if SCRIPT_DIR.name == "results(HPC)":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

RESULTS_INPUT_PATH = PROJECT_ROOT / "results(HPC)" / "results_crop_updated.csv"
PHYLUM_SAMPLES_INPUT_PATH = PROJECT_ROOT / "EDA" / "phylum_samples.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"

# If two taxonomy paths share no known level, the distance is undefined. I use
# NaN instead of assigning the maximum possible penalty (14), because there is no
# supported lowest common ancestor from the available data. If you prefer a hard
# penalty, set this to 14.0.
NO_SHARED_DISTANCE = np.nan

TOPK_PAIRWISE_LIMIT = 5
PD_DISPLAY_ROWS = 20


# %% [markdown]
# ## 2. Taxonomy Constants
#
# The hierarchy is encoded from broadest to narrowest. Species is index 6, so
# the LCA distance between two full species-level paths is:
#
# `distance = (6 - shared_level_index) + (6 - shared_level_index)`

# %%
TAXONOMIC_LEVELS = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]

LEVEL_TO_INDEX = {level: index for index, level in enumerate(TAXONOMIC_LEVELS)}
INDEX_TO_LEVEL = {index: level for level, index in LEVEL_TO_INDEX.items()}
SPECIES_LEVEL_INDEX = LEVEL_TO_INDEX["species"]

MODEL_TAXONOMY_COLUMNS = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]

USER_RESULTS_TAXONOMY_COLUMNS = [
    "user_kingdom",
    "user_phylum",
    "user_class",
    "user_order",
    "user_family",
    "user_genus",
    "User_classified",
]

PHYLUM_SAMPLE_TAXONOMY_COLUMNS = [
    "taxon_kingdom",
    "taxon_phylum",
    "taxon_class",
    "taxon_order",
    "taxon_family",
    "taxon_genus",
    "taxon_species",
]

PHYLUM_TO_RESULTS_USER_COLUMN = dict(
    zip(PHYLUM_SAMPLE_TAXONOMY_COLUMNS, USER_RESULTS_TAXONOMY_COLUMNS)
)

KNOWN_COLUMNS = {
    "crop_file",
    "crop_file_path",
    "image_id",
    "Falsely identified",
    "box_num",
    "species",
    "User_classified",
    "same_species?",
    "My analysis",
    "common_name",
    "kingdom",
    "user_kingdom",
    "phylum",
    "user_phylum",
    "class",
    "user_class",
    "order",
    "user_order",
    "family",
    "user_family",
    "genus",
    "user_genus",
    "score",
    "top_no",
    "topk_predictions_json",
    "observation_id",
    "photo_id",
    "Image_name",
    "scientific_name",
    "taxon_rank",
    "pred_prey",
    "special_type_of_feeding",
    *PHYLUM_SAMPLE_TAXONOMY_COLUMNS,
}

MISSING_STRING_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "<na>",
}

TRUE_STRINGS = {
    "true",
    "t",
    "yes",
    "y",
    "1",
    "same",
    "match",
    "matched",
    "correct",
    "exact",
    "exact match",
    "same species",
}

FALSE_STRINGS = {
    "false",
    "f",
    "no",
    "n",
    "0",
    "different",
    "diff",
    "mismatch",
    "not same",
    "not same species",
    "incorrect",
    "wrong",
}

# JSON blobs can be very large, and lowercasing them is unnecessary for these
# metrics. Taxonomic labels and IDs used in the analysis are normalized.
TEXT_NORMALIZATION_SKIP_COLUMNS = {"topk_predictions_json"}


# %% [markdown]
# ## 3. General Utilities

# %%
def print_warning(message: str) -> None:
    """Print a warning in a consistent format."""
    print(f"[WARNING] {message}")


def print_section(title: str) -> None:
    """Print a readable section header for script/notebook output."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def resolve_input_path(path: Path | str) -> Path:
    """
    Resolve a CSV/Excel input path.

    If a path without an extension is supplied, try `.csv`, `.xlsx`, and `.xls`.
    """
    path = Path(path)
    if path.exists():
        return path

    if path.suffix == "":
        for suffix in (".csv", ".xlsx", ".xls"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Input file does not exist: {path}")


def standardize_known_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names and standardize known columns by case.

    Example: `user_class ` becomes `user_class`; `image_name` becomes
    `Image_name` if that column appears with different capitalization.
    """
    df = df.copy()
    df.columns = [str(column).strip().lstrip("\ufeff") for column in df.columns]

    known_by_lower = {column.lower(): column for column in KNOWN_COLUMNS}
    existing_columns = set(df.columns)
    rename_map: dict[str, str] = {}

    for column in df.columns:
        canonical_column = known_by_lower.get(column.lower())
        if canonical_column is None or canonical_column == column:
            continue

        if canonical_column in existing_columns:
            print_warning(
                f"Both '{column}' and canonical '{canonical_column}' exist. "
                f"Leaving '{column}' unchanged to avoid duplicate columns."
            )
            continue

        rename_map[column] = canonical_column
        existing_columns.add(canonical_column)

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def normalize_taxon_value(value: object) -> str | float:
    """
    Normalize a taxonomic/string value for comparison.

    Returns lowercase stripped text, or `np.nan` for missing-like values. This
    function is used by the taxonomy path and distance functions, so comparisons
    are case-insensitive and robust to extra whitespace.
    """
    if pd.isna(value):
        return np.nan

    normalized = str(value).strip().lower()
    if normalized in MISSING_STRING_VALUES:
        return np.nan

    return normalized


def normalize_text_series(series: pd.Series) -> pd.Series:
    """Vectorized version of `normalize_taxon_value()` for dataframe columns."""
    normalized = series.astype("string").str.strip().str.lower()
    normalized = normalized.mask(normalized.isin(MISSING_STRING_VALUES))
    return normalized.astype(object).where(normalized.notna(), np.nan)


def normalize_dataframe_strings(
    df: pd.DataFrame,
    skip_columns: Iterable[str] = TEXT_NORMALIZATION_SKIP_COLUMNS,
) -> pd.DataFrame:
    """
    Normalize string-like columns by stripping whitespace, lowercasing, and
    converting missing-like values to NaN.

    `topk_predictions_json` is skipped by default because it is a large JSON
    blob and is not needed for these metrics.
    """
    df = df.copy()
    skip_columns = set(skip_columns)

    for column in df.columns:
        if column in skip_columns:
            continue

        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(
            df[column]
        ):
            df[column] = normalize_text_series(df[column])

    return df


def load_data_file(path: Path | str) -> pd.DataFrame:
    """
    Load a CSV or Excel file and apply standard column/value cleaning.

    All columns are initially read as strings so IDs such as image names and
    observation numbers are not accidentally converted or reformatted.
    """
    resolved_path = resolve_input_path(path)
    suffix = resolved_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(
            resolved_path,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            low_memory=False,
        )
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(
            resolved_path,
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {resolved_path}. "
            "Use CSV, XLSX, or XLS."
        )

    df = standardize_known_column_names(df)
    df = normalize_dataframe_strings(df)
    print(f"Loaded {resolved_path} with {len(df):,} rows and {len(df.columns):,} columns.")
    return df


def validate_columns(
    df: pd.DataFrame,
    expected_columns: Sequence[str],
    dataframe_name: str,
    required: bool = False,
) -> list[str]:
    """Check whether expected columns exist and print a clear warning if not."""
    missing_columns = [column for column in expected_columns if column not in df.columns]

    if missing_columns:
        severity = "required" if required else "expected"
        print_warning(
            f"{dataframe_name} is missing {severity} columns: "
            f"{', '.join(missing_columns)}"
        )

    return missing_columns


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Ensure columns exist. Missing columns are added as NaN so downstream logic can
    continue and produce partial metrics rather than failing with KeyError.
    """
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def parse_boolish(value: object) -> bool | float:
    """
    Parse inconsistent boolean values.

    Handles True/False, yes/no, same/different, 1/0, and similar labels. Returns
    np.nan when the value cannot be interpreted.
    """
    normalized = normalize_taxon_value(value)
    if pd.isna(normalized):
        return np.nan

    if normalized in TRUE_STRINGS:
        return True
    if normalized in FALSE_STRINGS:
        return False

    return np.nan


def bool_label(value: object) -> str:
    """Convert parsed boolean values to readable grouping labels."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values, coercing invalid entries to NaN."""
    return pd.to_numeric(series, errors="coerce")


def numeric_or_nan(value: object) -> float:
    """Convert one scalar to float, returning NaN when conversion is not possible."""
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return np.nan
    return float(numeric_value)


def safe_mean(values: Sequence[float]) -> float:
    """Mean that ignores NaN values and returns NaN when no values are valid."""
    valid_values = pd.Series(values, dtype="float64").dropna()
    if valid_values.empty:
        return np.nan
    return float(valid_values.mean())


def safe_min(values: Sequence[float]) -> float:
    """Minimum that ignores NaN values and returns NaN when no values are valid."""
    valid_values = pd.Series(values, dtype="float64").dropna()
    if valid_values.empty:
        return np.nan
    return float(valid_values.min())


def safe_max(values: Sequence[float]) -> float:
    """Maximum that ignores NaN values and returns NaN when no values are valid."""
    valid_values = pd.Series(values, dtype="float64").dropna()
    if valid_values.empty:
        return np.nan
    return float(valid_values.max())


def safe_std(values: Sequence[float]) -> float:
    """
    Population standard deviation that ignores NaN values.

    With one valid pairwise distance, the standard deviation is 0.0. With no
    valid pairwise distances, it is NaN.
    """
    valid_values = pd.Series(values, dtype="float64").dropna()
    if valid_values.empty:
        return np.nan
    return float(valid_values.std(ddof=0))


# %% [markdown]
# ## 4. Taxonomic Path and Distance Functions

# %%
def build_taxonomy_path(
    row: pd.Series | dict[str, object],
    columns: Sequence[str],
) -> list[str | float]:
    """
    Build a normalized taxonomy path from a row and a list of columns.

    Example model path:
    `[kingdom, phylum, class, order, family, genus, species]`

    Example user path:
    `[user_kingdom, user_phylum, user_class, user_order, user_family,
    user_genus, User_classified]`
    """
    return [normalize_taxon_value(row.get(column, np.nan)) for column in columns]


def is_known_taxon(value: object) -> bool:
    """Return True when a taxonomy value is present after normalization."""
    return not pd.isna(normalize_taxon_value(value))


def deepest_shared_taxonomic_level(
    path_a: Sequence[object],
    path_b: Sequence[object],
    return_index: bool = False,
) -> str | tuple[int | None, str]:
    """
    Find the deepest shared taxonomic level using LCA-style logic.

    The paths are compared from kingdom to species. If both values are known and
    different at a level, comparison stops because the taxa have diverged. Missing
    values do not create a match; they are skipped so that available lower-level
    evidence can still be used when the higher-level taxonomy is incomplete.

    Returns:
    - `"species"`, `"genus"`, ..., `"kingdom"` when a shared level is found.
    - `"none"` when no shared known level is found.

    If `return_index=True`, returns `(index, level_name)` where index is `None`
    when no shared known level exists.
    """
    deepest_index: int | None = None

    for level_index, (value_a, value_b) in enumerate(zip(path_a, path_b)):
        normalized_a = normalize_taxon_value(value_a)
        normalized_b = normalize_taxon_value(value_b)

        # Missing values cannot establish a shared level, but they also do not
        # prove divergence. Continue to lower levels in case the data is sparse.
        if pd.isna(normalized_a) or pd.isna(normalized_b):
            continue

        if normalized_a == normalized_b:
            deepest_index = level_index
        else:
            break

    deepest_level = INDEX_TO_LEVEL.get(deepest_index, "none")

    if return_index:
        return deepest_index, deepest_level
    return deepest_level


def taxonomic_distance(
    path_a: Sequence[object],
    path_b: Sequence[object],
    no_shared_distance: float = NO_SHARED_DISTANCE,
) -> float:
    """
    Calculate taxonomic distance between two taxonomy paths.

    Distance examples for full species-level paths:
    - same species: 0
    - same genus, different species: 2
    - same family, different genus/species: 4
    - same order only: 6
    - same class only: 8
    - same phylum only: 10
    - same kingdom only: 12
    - no shared known level: NaN by default, controlled by `no_shared_distance`
    """
    shared_index, _ = deepest_shared_taxonomic_level(
        path_a,
        path_b,
        return_index=True,
    )

    if shared_index is None:
        return no_shared_distance

    return float((SPECIES_LEVEL_INDEX - shared_index) * 2)


# %% [markdown]
# ## 5. Fill/Verify User Taxonomy from `phylum_samples`
#
# The taxonomy already in `results_crop_updated` is preferred. The
# `phylum_samples` taxonomy is used only to fill missing user-taxonomy fields and
# to report conflicts where both sources are present but disagree.

# %%
def fill_user_taxonomy_from_phylum_samples(
    results_df: pd.DataFrame,
    phylum_samples_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge user taxonomy from phylum_samples by image key and fill missing values.

    `phylum_samples.Image_name` corresponds to `results_crop_updated.image_id`.
    Existing user taxonomy in the results file is preserved.
    """
    results_df = results_df.copy()

    validate_columns(
        results_df,
        ["image_id", *USER_RESULTS_TAXONOMY_COLUMNS],
        "results_crop_updated",
    )
    results_df = ensure_columns(results_df, USER_RESULTS_TAXONOMY_COLUMNS)

    if phylum_samples_df is None:
        print_warning("phylum_samples was not loaded; using only results user taxonomy.")
        return results_df

    missing_phylum_columns = validate_columns(
        phylum_samples_df,
        ["Image_name", *PHYLUM_SAMPLE_TAXONOMY_COLUMNS],
        "phylum_samples",
    )
    if "image_id" not in results_df.columns or "Image_name" in missing_phylum_columns:
        print_warning(
            "Cannot merge phylum_samples because image_id/Image_name is missing. "
            "Using only user taxonomy already present in results."
        )
        return results_df

    available_phylum_columns = [
        column
        for column in ["Image_name", *PHYLUM_SAMPLE_TAXONOMY_COLUMNS]
        if column in phylum_samples_df.columns
    ]

    samples = phylum_samples_df[available_phylum_columns].copy()
    samples = samples.dropna(subset=["Image_name"])

    duplicate_image_names = int(samples["Image_name"].duplicated().sum())
    if duplicate_image_names:
        print_warning(
            f"phylum_samples has {duplicate_image_names:,} duplicate Image_name rows. "
            "Keeping the first row for each Image_name during merge."
        )

    samples = samples.drop_duplicates(subset=["Image_name"], keep="first")

    sample_rename_map = {
        column: f"phylum_sample_{column}" for column in available_phylum_columns
    }
    samples = samples.rename(columns=sample_rename_map)

    merged = results_df.merge(
        samples,
        how="left",
        left_on="image_id",
        right_on="phylum_sample_Image_name",
    )

    unmatched_rows = int(merged["phylum_sample_Image_name"].isna().sum())
    if unmatched_rows:
        print_warning(
            f"{unmatched_rows:,} result rows did not find a matching Image_name in "
            "phylum_samples."
        )

    fill_records = []
    conflict_records = []

    for phylum_column, user_column in PHYLUM_TO_RESULTS_USER_COLUMN.items():
        sample_column = f"phylum_sample_{phylum_column}"
        if sample_column not in merged.columns:
            continue

        existing_user = merged[user_column]
        sample_user = merged[sample_column]

        conflict_mask = (
            existing_user.notna()
            & sample_user.notna()
            & (existing_user != sample_user)
        )
        fill_mask = existing_user.isna() & sample_user.notna()

        conflict_count = int(conflict_mask.sum())
        fill_count = int(fill_mask.sum())

        if fill_count:
            merged.loc[fill_mask, user_column] = merged.loc[fill_mask, sample_column]

        fill_records.append(
            {
                "user_column": user_column,
                "phylum_sample_column": phylum_column,
                "filled_missing_rows": fill_count,
            }
        )
        conflict_records.append(
            {
                "user_column": user_column,
                "phylum_sample_column": phylum_column,
                "conflicting_non_missing_rows": conflict_count,
            }
        )

    fill_summary = pd.DataFrame(fill_records)
    conflict_summary = pd.DataFrame(conflict_records)

    if not fill_summary.empty and fill_summary["filled_missing_rows"].sum() > 0:
        print("\nFilled missing user taxonomy from phylum_samples:")
        print(fill_summary.to_string(index=False))

    if (
        not conflict_summary.empty
        and conflict_summary["conflicting_non_missing_rows"].sum() > 0
    ):
        print("\nUser-taxonomy conflicts between results and phylum_samples:")
        print(conflict_summary.to_string(index=False))

    return merged


# %% [markdown]
# ## 6. Metric A: Prediction-Level Distance to Observer Label

# %%
def add_prediction_level_taxonomic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-level taxonomic distance and shared-level indicators.

    This uses the taxonomy columns in `results_crop_updated` after missing user
    taxonomy has optionally been filled from `phylum_samples`.
    """
    df = df.copy()

    all_taxonomy_columns = [*MODEL_TAXONOMY_COLUMNS, *USER_RESULTS_TAXONOMY_COLUMNS]
    missing_taxonomy_columns = validate_columns(
        df,
        all_taxonomy_columns,
        "analysis dataframe",
    )
    if missing_taxonomy_columns:
        df = ensure_columns(df, missing_taxonomy_columns)

    if "same_species?" not in df.columns:
        print_warning("same_species? is missing; exact match will use species labels only.")
        df["same_species?"] = np.nan

    if "top_no" not in df.columns:
        print_warning("top_no is missing; rank-based summaries will be incomplete.")
        df["top_no"] = np.nan

    if "score" not in df.columns:
        print_warning("score is missing; score-gap summaries will be incomplete.")
        df["score"] = np.nan

    df["top_no_numeric"] = safe_numeric(df["top_no"])
    df["score_numeric"] = safe_numeric(df["score"])
    df["same_species_bool"] = df["same_species?"].map(parse_boolish)

    shared_index = np.full(len(df), -1, dtype=int)
    comparison_active = np.ones(len(df), dtype=bool)

    for level_index, (model_column, user_column) in enumerate(
        zip(MODEL_TAXONOMY_COLUMNS, USER_RESULTS_TAXONOMY_COLUMNS)
    ):
        model_values = df[model_column]
        user_values = df[user_column]

        both_known = model_values.notna() & user_values.notna()
        equal_here = comparison_active & both_known & (model_values == user_values)
        mismatch_here = comparison_active & both_known & (model_values != user_values)

        shared_index[equal_here.to_numpy()] = level_index
        comparison_active[mismatch_here.to_numpy()] = False

    # Exact species matches can be identified from the species labels and/or the
    # user-provided same_species? flag. When either says exact match, the LCA is
    # species and the taxonomic distance is 0.
    species_equal = (
        df["species"].notna()
        & df["User_classified"].notna()
        & (df["species"] == df["User_classified"])
    )
    same_species_true = df["same_species_bool"].map(lambda value: value is True)
    exact_species_mask = species_equal | same_species_true
    shared_index[exact_species_mask.to_numpy()] = SPECIES_LEVEL_INDEX

    distance_values = np.where(
        shared_index >= 0,
        (SPECIES_LEVEL_INDEX - shared_index) * 2,
        NO_SHARED_DISTANCE,
    ).astype(float)

    df["taxonomic_distance_to_user"] = distance_values
    df["deepest_shared_level_with_user"] = [
        INDEX_TO_LEVEL.get(index, "none") for index in shared_index
    ]

    # These are cumulative match indicators. Example: if taxa match at family,
    # then kingdom/phylum/class/order/family are True, while genus/species are
    # False.
    df["is_kingdom_match"] = shared_index >= LEVEL_TO_INDEX["kingdom"]
    df["is_phylum_match"] = shared_index >= LEVEL_TO_INDEX["phylum"]
    df["is_class_match"] = shared_index >= LEVEL_TO_INDEX["class"]
    df["is_order_match"] = shared_index >= LEVEL_TO_INDEX["order"]
    df["is_family_match"] = shared_index >= LEVEL_TO_INDEX["family"]
    df["is_genus_match"] = shared_index >= LEVEL_TO_INDEX["genus"]
    df["is_exact_species_match"] = shared_index >= LEVEL_TO_INDEX["species"]

    # Extra helper used by Metric C. It follows the requested rule: correct if
    # same_species? is true OR the predicted species equals the observer label.
    df["is_prediction_correct"] = exact_species_mask

    return df


def categorize_error_near_miss(deepest_shared_level: object) -> str:
    """
    Categorize non-exact predictions into near-miss or distant-error buckets.
    """
    level = normalize_taxon_value(deepest_shared_level)

    if level == "genus":
        return "same genus but different species"
    if level == "family":
        return "same family but different genus"
    if level == "order":
        return "same order but different family"

    return "taxonomically distant errors"


def summarize_prediction_level_distances(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Create aggregate summaries for prediction-level taxonomic distances.

    Returns:
    - a combined long-ish dataframe suitable for saving as one CSV
    - a dictionary of readable summary tables for printing
    """
    df = df.copy()
    df["taxonomic_distance_to_user"] = safe_numeric(
        df["taxonomic_distance_to_user"]
    )

    total_predictions = len(df)
    valid_distance_count = int(df["taxonomic_distance_to_user"].notna().sum())

    overall_summary = pd.DataFrame(
        [
            {
                "summary_table": "overall",
                "prediction_count": total_predictions,
                "valid_distance_count": valid_distance_count,
                "average_taxonomic_distance": df[
                    "taxonomic_distance_to_user"
                ].mean(),
            }
        ]
    )

    if "top_no_numeric" in df.columns:
        by_top_no = (
            df.groupby("top_no_numeric", dropna=False)
            .agg(
                prediction_count=("taxonomic_distance_to_user", "size"),
                valid_distance_count=("taxonomic_distance_to_user", "count"),
                average_taxonomic_distance=("taxonomic_distance_to_user", "mean"),
            )
            .reset_index()
            .rename(columns={"top_no_numeric": "top_no"})
            .sort_values("top_no", na_position="last")
        )
        by_top_no.insert(0, "summary_table", "average_distance_by_top_no")
    else:
        by_top_no = pd.DataFrame()

    if "same_species_bool" in df.columns:
        df["same_species_label"] = df["same_species_bool"].map(bool_label)
    else:
        df["same_species_label"] = "unknown"

    by_same_species = (
        df.groupby("same_species_label", dropna=False)
        .agg(
            prediction_count=("taxonomic_distance_to_user", "size"),
            valid_distance_count=("taxonomic_distance_to_user", "count"),
            average_taxonomic_distance=("taxonomic_distance_to_user", "mean"),
        )
        .reset_index()
        .rename(columns={"same_species_label": "same_species"})
    )
    by_same_species.insert(0, "summary_table", "average_distance_by_same_species")

    level_order = [
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
        "none",
    ]
    deepest_counts = (
        df["deepest_shared_level_with_user"]
        .fillna("none")
        .value_counts()
        .reindex(level_order, fill_value=0)
        .rename_axis("deepest_shared_level_with_user")
        .reset_index(name="prediction_count")
    )
    if total_predictions:
        deepest_counts["percentage_of_predictions"] = (
            deepest_counts["prediction_count"] / total_predictions * 100
        )
    else:
        deepest_counts["percentage_of_predictions"] = np.nan
    deepest_counts.insert(0, "summary_table", "deepest_shared_level_distribution")

    errors = df.loc[~df["is_exact_species_match"].fillna(False)].copy()
    errors["near_miss_category"] = errors[
        "deepest_shared_level_with_user"
    ].map(categorize_error_near_miss)

    near_miss_order = [
        "same genus but different species",
        "same family but different genus",
        "same order but different family",
        "taxonomically distant errors",
    ]
    near_miss_counts = (
        errors["near_miss_category"]
        .value_counts()
        .reindex(near_miss_order, fill_value=0)
        .rename_axis("near_miss_category")
        .reset_index(name="error_count")
    )
    if len(errors):
        near_miss_counts["percentage_of_errors"] = (
            near_miss_counts["error_count"] / len(errors) * 100
        )
    else:
        near_miss_counts["percentage_of_errors"] = np.nan
    near_miss_counts["total_error_count"] = len(errors)
    near_miss_counts.insert(0, "summary_table", "near_miss_error_distribution")

    aggregate_summary = pd.concat(
        [
            overall_summary,
            by_top_no,
            by_same_species,
            deepest_counts,
            near_miss_counts,
        ],
        ignore_index=True,
        sort=False,
    )

    summary_tables = {
        "overall": overall_summary,
        "average_distance_by_top_no": by_top_no,
        "average_distance_by_same_species": by_same_species,
        "deepest_shared_level_distribution": deepest_counts,
        "near_miss_error_distribution": near_miss_counts,
    }

    return aggregate_summary, summary_tables


# %% [markdown]
# ## 7. Metric B: Top-5 Pairwise Confusion Within Each Image/Crop

# %%
def get_group_columns(df: pd.DataFrame) -> list[str]:
    """
    Choose grouping columns for crop-level metrics.

    Use `image_id + box_num` when `box_num` exists; otherwise use `image_id`.
    """
    if "image_id" not in df.columns:
        print_warning("image_id is missing; crop-level summaries cannot be computed.")
        return []

    group_columns = ["image_id"]
    if "box_num" in df.columns:
        group_columns.append("box_num")

    return group_columns


def first_rank_row(group: pd.DataFrame, rank: int) -> pd.Series | None:
    """Return the first row for a requested top-k rank, or None if absent."""
    rank_rows = group.loc[group["top_no_numeric"] == rank]
    if rank_rows.empty:
        return None
    return rank_rows.iloc[0]


def distance_between_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    taxonomy_columns: Sequence[str] = MODEL_TAXONOMY_COLUMNS,
) -> float:
    """Taxonomic distance between two rows using the requested taxonomy columns."""
    path_a = build_taxonomy_path(row_a, taxonomy_columns)
    path_b = build_taxonomy_path(row_b, taxonomy_columns)
    return taxonomic_distance(path_a, path_b)


def summarize_topk_pairwise_confusion(
    df: pd.DataFrame,
    topk_limit: int = TOPK_PAIRWISE_LIMIT,
) -> pd.DataFrame:
    """
    Summarize pairwise taxonomic distances among top-k model predictions.

    One row is produced for each image/crop group. The function does not assume
    every group has exactly five predictions.
    """
    group_columns = get_group_columns(df)
    if not group_columns:
        return pd.DataFrame()

    if "top_no_numeric" not in df.columns:
        print_warning("top_no_numeric is missing; cannot identify top-k predictions.")
        return pd.DataFrame()

    if "score_numeric" not in df.columns:
        print_warning("score_numeric is missing; score-gap metrics will be NaN.")
        df = df.copy()
        df["score_numeric"] = np.nan

    topk_df = df.loc[
        df["top_no_numeric"].between(1, topk_limit, inclusive="both")
    ].copy()

    if topk_df.empty:
        print_warning(f"No predictions found with top_no <= {topk_limit}.")
        return pd.DataFrame()

    sort_columns = [*group_columns, "top_no_numeric", "score_numeric"]
    topk_df = topk_df.sort_values(
        sort_columns,
        ascending=[True] * (len(sort_columns) - 1) + [False],
        na_position="last",
    )

    records = []
    grouped = topk_df.groupby(group_columns, dropna=False, sort=False)

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        record = dict(zip(group_columns, group_key))
        group = group.sort_values(
            ["top_no_numeric", "score_numeric"],
            ascending=[True, False],
            na_position="last",
        )

        row_records = [row for _, row in group.iterrows()]
        paths = [build_taxonomy_path(row, MODEL_TAXONOMY_COLUMNS) for row in row_records]

        pairwise_distances = [
            taxonomic_distance(paths[index_a], paths[index_b])
            for index_a, index_b in combinations(range(len(paths)), 2)
        ]

        top1 = first_rank_row(group, 1)
        top2 = first_rank_row(group, 2)
        top5 = first_rank_row(group, 5)

        top1_score = numeric_or_nan(top1["score_numeric"]) if top1 is not None else np.nan
        top2_score = numeric_or_nan(top2["score_numeric"]) if top2 is not None else np.nan
        top5_score = numeric_or_nan(top5["score_numeric"]) if top5 is not None else np.nan

        top1_to_top2_distance = (
            distance_between_rows(top1, top2)
            if top1 is not None and top2 is not None
            else np.nan
        )

        if top1 is not None:
            top1_distances = [
                distance_between_rows(top1, other_row)
                for other_row in row_records
                if other_row.name != top1.name
            ]
        else:
            top1_distances = []

        record.update(
            {
                "number_of_topk_predictions": int(len(group)),
                "mean_pairwise_top5_taxonomic_distance": safe_mean(pairwise_distances),
                "min_pairwise_top5_taxonomic_distance": safe_min(pairwise_distances),
                "max_pairwise_top5_taxonomic_distance": safe_max(pairwise_distances),
                "std_pairwise_top5_taxonomic_distance": safe_std(pairwise_distances),
                "top1_to_top2_taxonomic_distance": top1_to_top2_distance,
                "top1_to_top5_mean_taxonomic_distance": safe_mean(top1_distances),
                "top1_score": top1_score,
                "top2_score": top2_score,
                "score_gap_top1_top2": (
                    top1_score - top2_score
                    if pd.notna(top1_score) and pd.notna(top2_score)
                    else np.nan
                ),
                "score_gap_top1_top5": (
                    top1_score - top5_score
                    if pd.notna(top1_score) and pd.notna(top5_score)
                    else np.nan
                ),
            }
        )

        records.append(record)

    return pd.DataFrame(records)


# %% [markdown]
# ## 8. Metric C: Correct Species Present Somewhere in Top-k

# %%
def summarize_correct_in_topk_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize groups where at least one top-k prediction is correct.

    Correct means `same_species? == True` or `species == User_classified` after
    normalization. For those groups, this calculates how taxonomically close the
    incorrect top-k predictions are to the observer label.
    """
    group_columns = get_group_columns(df)
    if not group_columns:
        return pd.DataFrame()

    required_columns = [
        "is_prediction_correct",
        "taxonomic_distance_to_user",
        "top_no_numeric",
    ]
    missing_columns = validate_columns(df, required_columns, "analysis dataframe")
    if missing_columns:
        print_warning(
            "Correct-in-top-k summary cannot be computed without prediction-level "
            "metric columns."
        )
        return pd.DataFrame()

    working_df = df.copy()
    working_df["taxonomic_distance_to_user"] = safe_numeric(
        working_df["taxonomic_distance_to_user"]
    )

    sort_columns = [*group_columns, "top_no_numeric"]
    working_df = working_df.sort_values(sort_columns, na_position="last")

    records = []
    grouped = working_df.groupby(group_columns, dropna=False, sort=False)

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        correct_mask = group["is_prediction_correct"].fillna(False).astype(bool)
        if not correct_mask.any():
            continue

        correct_rows = group.loc[correct_mask]
        incorrect_rows = group.loc[~correct_mask]
        incorrect_distances = incorrect_rows["taxonomic_distance_to_user"].dropna()

        correct_ranks = correct_rows["top_no_numeric"].dropna()
        first_correct_rank = (
            float(correct_ranks.min()) if not correct_ranks.empty else np.nan
        )

        record = dict(zip(group_columns, group_key))
        record.update(
            {
                "has_correct_in_topk": True,
                "first_correct_rank": first_correct_rank,
                "number_of_topk_predictions": int(len(group)),
                "number_correct_in_topk": int(correct_mask.sum()),
                "mean_distance_to_user_among_incorrect_topk": (
                    float(incorrect_distances.mean())
                    if not incorrect_distances.empty
                    else np.nan
                ),
                "min_distance_to_user_among_incorrect_topk": (
                    float(incorrect_distances.min())
                    if not incorrect_distances.empty
                    else np.nan
                ),
                "max_distance_to_user_among_incorrect_topk": (
                    float(incorrect_distances.max())
                    if not incorrect_distances.empty
                    else np.nan
                ),
            }
        )
        records.append(record)

    return pd.DataFrame(records)


# %% [markdown]
# ## 9. Output and Reporting Helpers

# %%
def save_outputs(
    prediction_level_df: pd.DataFrame,
    top5_pairwise_df: pd.DataFrame,
    correct_in_topk_df: pd.DataFrame,
    aggregate_summary_df: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
) -> None:
    """Save all requested outputs into `analysis_outputs/`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "prediction_level_taxonomic_distances.csv": prediction_level_df,
        "top5_pairwise_confusion_by_crop.csv": top5_pairwise_df,
        "topk_correct_present_distance_summary.csv": correct_in_topk_df,
        "aggregate_taxonomic_distance_summary.csv": aggregate_summary_df,
    }

    for filename, dataframe in output_paths.items():
        output_path = output_dir / filename
        dataframe.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved {output_path} ({len(dataframe):,} rows).")


def print_summary_tables(
    summary_tables: dict[str, pd.DataFrame],
    top5_pairwise_df: pd.DataFrame,
    correct_in_topk_df: pd.DataFrame,
) -> None:
    """Print readable summary tables for notebook/script use."""
    with pd.option_context(
        "display.max_rows",
        PD_DISPLAY_ROWS,
        "display.max_columns",
        None,
        "display.width",
        160,
    ):
        print_section("Metric A: Overall Taxonomic Distance")
        print(summary_tables["overall"].to_string(index=False))

        print_section("Metric A: Average Distance by top_no")
        print(
            summary_tables["average_distance_by_top_no"]
            .head(PD_DISPLAY_ROWS)
            .to_string(index=False)
        )

        print_section("Metric A: Average Distance by same_species?")
        print(
            summary_tables["average_distance_by_same_species"].to_string(index=False)
        )

        print_section("Metric A: Deepest Shared Level Distribution")
        print(
            summary_tables["deepest_shared_level_distribution"].to_string(index=False)
        )

        print_section("Metric A: Near-Miss Error Distribution")
        print(summary_tables["near_miss_error_distribution"].to_string(index=False))

        print_section("Metric B: Top-5 Pairwise Confusion Preview")
        if top5_pairwise_df.empty:
            print("No top-5 pairwise summary rows were produced.")
        else:
            print(top5_pairwise_df.head(PD_DISPLAY_ROWS).to_string(index=False))

        print_section("Metric C: Correct-in-Top-k Preview")
        if correct_in_topk_df.empty:
            print("No groups contained a correct prediction in top-k.")
        else:
            print(correct_in_topk_df.head(PD_DISPLAY_ROWS).to_string(index=False))


def print_interpretation_notes() -> None:
    """Print a concise interpretation guide for the generated metrics."""
    print_section("Interpretation Guide")
    print(
        "Distance 0 means exact species match.\n"
        "Distance 2 means same genus but different species.\n"
        "Distance 4 means same family but different genus.\n"
        "Larger distances indicate broader taxonomic disagreement.\n"
        "Low top-5 pairwise distance means top predictions are biologically close.\n"
        "High top-5 pairwise distance means model uncertainty is taxonomically broad.\n"
        "No shared known level is written as NaN by default; set NO_SHARED_DISTANCE = 14.0 "
        "if you prefer a maximum-distance penalty."
    )


# %% [markdown]
# ## 10. Run the Analysis

# %%
def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete taxonomic confusion analysis."""
    print_section("Loading Inputs")
    results_df = load_data_file(RESULTS_INPUT_PATH)

    try:
        phylum_samples_df = load_data_file(PHYLUM_SAMPLES_INPUT_PATH)
    except FileNotFoundError:
        print_warning(f"Could not find phylum_samples at {PHYLUM_SAMPLES_INPUT_PATH}.")
        phylum_samples_df = None

    print_section("Preparing Analysis Data")
    results_with_filled_user_taxonomy = fill_user_taxonomy_from_phylum_samples(
        results_df,
        phylum_samples_df,
    )
    prediction_level_df = add_prediction_level_taxonomic_metrics(
        results_with_filled_user_taxonomy
    )

    print_section("Computing Aggregate Summaries")
    aggregate_summary_df, summary_tables = summarize_prediction_level_distances(
        prediction_level_df
    )

    print_section("Computing Top-5 Pairwise Confusion")
    top5_pairwise_df = summarize_topk_pairwise_confusion(prediction_level_df)

    print_section("Computing Correct-in-Top-k Cases")
    correct_in_topk_df = summarize_correct_in_topk_cases(prediction_level_df)

    print_section("Saving Outputs")
    save_outputs(
        prediction_level_df=prediction_level_df,
        top5_pairwise_df=top5_pairwise_df,
        correct_in_topk_df=correct_in_topk_df,
        aggregate_summary_df=aggregate_summary_df,
        output_dir=OUTPUT_DIR,
    )

    print_summary_tables(
        summary_tables=summary_tables,
        top5_pairwise_df=top5_pairwise_df,
        correct_in_topk_df=correct_in_topk_df,
    )
    print_interpretation_notes()

    return (
        prediction_level_df,
        top5_pairwise_df,
        correct_in_topk_df,
        aggregate_summary_df,
    )


if __name__ == "__main__":
    (
        prediction_level_taxonomic_distances,
        top5_pairwise_confusion_by_crop,
        topk_correct_present_distance_summary,
        aggregate_taxonomic_distance_summary,
    ) = main()

import pandas as pd
import numpy as np

import transformers
import torch
import accelerate

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model_id = "/share/ftrscape/lmiddha/models/llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16},
    temperature=0.3,
    top_p=0.9,
    truncation=True,
    max_length=500,
)

df = pd.read_csv("Dataset.csv")
print(df.head())

SYSTEM_PROMPT = """
You are generating a concise detection prompt for Grounding DINO. 

TASK:
Create 1-3 short object detection phrases to help locate a species in an image. 

RULES:
- Output ONLY the detection prompt text.
- Use short noun phrases, NOT full sentences.
- Focus on visible traits (color, shape, body type, wings, fins, leaves, petals, spores, etc.).
- Include 1-2 higher-level taxonomic descriptors when helpful (e.g., bird, insect, flowering plant, fungus).
- Avoid vague words like "animal" or "organism" unless no better descriptor exists.
- Do NOT include explanations.
- Always add a fullstop (.) at the end of the last phrase

EXAMPLES:

# Example 1: bird
input:
scientific_name: Ardea alba
common_name: Great Egret
taxon_class: Aves
Expected Prompt:
white bird with long legs, slender neck.

# Example 2: insect
input:
scientific_name: Danaus plexippus
common_name: Monarch Butterfly
taxon_class: Insecta
Expected Prompt:
orange butterfly with black patterned wings, monarch butterfly.

# Example 3: mammal
input:
scientific_name: Vulpes vulpes
common_name: Red Fox
taxon_class: Mammalia
Expected Prompt:
red fox with bushy tail, small carnivorous mammal.

# Example 4: flowering plant
input:
scientific_name: Rosa rubiginosa
common_name: Sweet Briar
taxon_class: Magnoliopsida
Expected Prompt:
pink flowering shrub, rose with thorny stems

# Example 5: fungus
input:
scientific_name: Amanita muscaria
common_name: Fly Agaric
taxon_class: Agaricomycetes
Expected Prompt:
red mushroom with white spots, cap-shaped fungus.
"""


def _safe_str(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()


def build_user_prompt(row: pd.Series) -> str:
    """Build the user prompt from a dataframe row."""
    return f"""
scientific_name: {_safe_str(row.get('scientific_name'))}
common_name: {_safe_str(row.get('common_name'))}
taxon_rank: {_safe_str(row.get('taxon_rank'))}
pred_prey: {_safe_str(row.get('pred_prey'))}
special_type_of_feeding: {_safe_str(row.get('special_type_of_feeding'))}
taxon_kingdom: {_safe_str(row.get('taxon_kingdom'))}
taxon_phylum: {_safe_str(row.get('taxon_phylum'))}
taxon_class: {_safe_str(row.get('taxon_class'))}
taxon_order: {_safe_str(row.get('taxon_order'))}
taxon_family: {_safe_str(row.get('taxon_family'))}
taxon_genus: {_safe_str(row.get('taxon_genus'))}
taxon_species: {_safe_str(row.get('taxon_species'))}

Generate the Grounding DINO detection prompt.
"""


def generate_prompt_for_row(row: pd.Series) -> str:
    """Call the LLM pipeline and return the generated prompt text for this row."""
    user_prompt = build_user_prompt(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    outputs = pipeline(messages, max_new_tokens=500)
    # x = outputs[0]["generated_text"][-1]["content"]
    generated = outputs[0]["generated_text"][-1]["content"]
    return generated.strip()


# Iterate through the dataframe, generate prompt for each row, add new column
grounding_dino_prompts = []
for idx, row in df.iterrows():
    x = generate_prompt_for_row(row)
    grounding_dino_prompts.append(x)
    print(f"Processed {idx}: prompt as: {x}")

df["grounding_dino_prompt"] = grounding_dino_prompts

print(df[["common_name", "scientific_name", "grounding_dino_prompt"]].head(10))

df.to_csv("/share/ftrscape/lmiddha/dataset/dataset_with_prompts_a100.csv", index=False)

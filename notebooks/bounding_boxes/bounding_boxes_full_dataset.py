import pandas as pd
import numpy as np
import os
from pathlib import Path
from PIL import Image 
from PIL import ImageOps
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import transformers
import torch
from IPython.display import display

df = pd.read_csv('/rs1/researchers/a/amallav/prompts/dataset_with_prompts_a100.csv')
print(df.head())

directory = "/rs1/researchers/a/amallav/image_dataset"
image_list = os.listdir(directory)
print(image_list)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_id = "/share/ftrscape/lmiddha/models/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def plot_boxes(results, image, top_k, save_path):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    sorted_indices = scores.argsort(descending=True)  # PyTorch tensor
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    sorted_indices_list = sorted_indices.tolist()
    labels = [labels[i] for i in sorted_indices_list]

    # Only take top_k
    boxes = boxes[:top_k]
    scores = scores[:top_k]
    labels = labels[:top_k]
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
    
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    
        ax.text(
            x1,
            y1,
            f"{label}: {score:.2f}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.5),
        )
    
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    return boxes, scores, labels

def detect_boxes(image, text, threshold:int, text_threshold:int):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=threshold,
    text_threshold=text_threshold,
    target_sizes=[image.size[::-1]]
    )

    return results

def save_csv(result_dir, boxes, scores, labels, image_path, observation_id, photo_id):
    output_file_bb = os.path.join(result_dir, "bounding_boxes_full_dataset.csv")
    rows = []
    ranks = list(range(1, len(scores) + 1))
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        rows.append({
            "observation_id": observation_id,
            "photo_id": photo_id,
            "image_path": image_path,
            "label": label,
            "score": float(score),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    df_bb = pd.DataFrame(rows)
    if os.path.exists(output_file_bb):
        df_bb.to_csv(output_file_bb, mode='a', header=False, index=False)
    else:
        df_bb.to_csv(output_file_bb, index=False)
    print("Successfully written bounding boxes csv")

    output_file_summary = os.path.join(result_dir, "summary_full_dataset.csv")
    summary_row = {
        "observation_id": observation_id,
        "photo_id": photo_id,
        "image_path": image_path,
        "num_boxes": len(boxes),
        "avg_score": float(scores.mean()) if len(scores) > 0 else 0.0,
        "max_score": float(scores.max()) if len(scores) > 0 else 0.0
    }
    df_summary = pd.DataFrame([summary_row])
    if os.path.exists(output_file_summary):
        df_summary.to_csv(output_file_summary, mode='a', header=False, index=False)
    else:
        df_summary.to_csv(output_file_summary, index=False)
    print("Successfully written summary csv")
    

def run_detections(data_dir, save_image_path, save_csv_path, threshold, text_threshold, top_k):
    for image in image_list:
        temp = image.split("_")
        observation_id = temp[1]
        photo_id = temp[3].split(".")[0]
        image_path = data_dir + image
        image_to_plot = Image.open(image_path).convert("RGB")
        match = df.loc[(df["observation_id"] == int(observation_id)) & (df["photo_id"] == int(photo_id))]
        prompt = match.iloc[0]["grounding_dino_prompt"]
        print(f"Image Path {image_path}, image prompt {prompt}")
        results = detect_boxes(image_to_plot, prompt, threshold, text_threshold)
        save_image_path_new = save_image_path + "obs_" + str(observation_id) + "_photo_" + str(photo_id) + ".png"
        boxes, scores, labels = plot_boxes(results, image_to_plot, top_k, save_image_path_new)
        save_csv(save_csv_path, boxes, scores, labels, image_path, observation_id, photo_id)


data_dir = "/rs1/researchers/a/amallav/image_dataset/"
save_image_path = "/rs1/researchers/a/amallav/results/outputs_gd/temp_images/"
save_csv_path = "/rs1/researchers/a/amallav/results/outputs_gd/"
run_detections(data_dir, save_image_path, save_csv_path, 0.2, 0.15, 5)        

# -----------------------------------------------------------
# IMAGE SEARCH SYSTEM (with YOLO object detection + CLIP embeddings)
# -----------------------------------------------------------
# This script allows you to:
# 1. Detect and crop objects from images using YOLO
# 2. Represent each cropped object as an embedding (numerical vector) using CLIP
# 3. Save all embeddings in a database (cached for reuse)
# 4. Search similar objects in the database using cosine similarity
# 5. Visualize query image vs. search results
# -----------------------------------------------------------

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import pickle
from matplotlib import pyplot as plt
import math
from ultralytics import YOLO

# -----------------------------------------------------------
# Step 1: DEVICE SETUP
# -----------------------------------------------------------
# Check if GPU is available. If yes, use "cuda" (faster). Otherwise, fallback to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
# Step 2: FOLDER STRUCTURE
# -----------------------------------------------------------
# IMAGE_FOLDER   -> contains all images we want to index into our database
# QUERY_FOLDER   -> contains images we want to use as queries
# CACHE_FILE     -> stores processed results so we don’t rebuild every time
IMAGE_FOLDER = "images"
QUERY_FOLDER = "query_images"
CACHE_FILE = "objects_cache.pkl"

print(f"Using device: {DEVICE}")

# -----------------------------------------------------------
# Step 3: LOAD MODELS
# -----------------------------------------------------------
# We need:
# - CLIP (for embeddings = understanding visual meaning)
# - YOLO (for object detection = finding objects inside images)
print("Loading models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
yolo_model = YOLO('yolov8n.pt')  # YOLOv8n = "nano" (very small, faster, less accurate)

# -----------------------------------------------------------
# Step 4: FUNCTION → Get CLIP Embedding
# -----------------------------------------------------------
def get_embedding(image_path_or_pil):
    """
    Convert an image (file path or PIL image) into a CLIP embedding vector.
    The embedding is normalized so comparisons work better.
    """
    # if input is a file path → open image
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        # if already a PIL image
        image = image_path_or_pil.convert("RGB")
    
    # convert to tensor and send to GPU/CPU
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)  # get CLIP features
    
    # normalize vector (important for cosine similarity)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

# -----------------------------------------------------------
# Step 5: FUNCTION → Detect + Crop Objects using YOLO
# -----------------------------------------------------------
def detect_and_crop_objects(image_path):
    """
    Use YOLO to detect objects inside an image.
    Each detected object is cropped out and stored.
    """
    results = yolo_model(image_path, conf=0.3)  # detect objects with confidence ≥ 0.3
    objects = []
    
    image = Image.open(image_path).convert("RGB")
    
    for result in results:
        if result.boxes is not None:  # if YOLO found anything
            for box in result.boxes:
                # extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_name = yolo_model.names[int(box.cls[0])]  # object label (car, dog, etc.)
                confidence = box.conf[0].cpu().numpy()          # detection confidence score
                
                # add padding around box (to avoid tight crop)
                padding = 10
                x1, y1 = max(0, int(x1-padding)), max(0, int(y1-padding))
                x2, y2 = min(image.width, int(x2+padding)), min(image.height, int(y2+padding))
                
                # crop out detected object
                cropped = image.crop((x1, y1, x2, y2))
                
                # save details of object
                objects.append({
                    'cropped_image': cropped,
                    'class': class_name,
                    'confidence': float(confidence),
                    'source_file': os.path.basename(image_path)
                })
    
    return objects

# -----------------------------------------------------------
# Step 6: FUNCTION → Build Database of Objects
# -----------------------------------------------------------
def build_database():
    """
    Process all images in IMAGE_FOLDER, detect objects, create embeddings,
    and store them in a database (cache for faster reuse).
    """
    # if cache already exists, load instead of recomputing
    if os.path.exists(CACHE_FILE):
        print("Loading cached data...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    
    print("Building database...")
    database = []
    
    # collect all images
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for i, filename in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {filename}")
        img_path = os.path.join(IMAGE_FOLDER, filename)
        
        # detect objects
        objects = detect_and_crop_objects(img_path)
        
        # create embeddings for each detected object
        for obj in objects:
            embedding = get_embedding(obj['cropped_image'])
            if embedding is not None:
                obj['embedding'] = embedding
                database.append(obj)
    
    # save database to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(database, f)
    
    print(f"Saved {len(database)} objects!")
    return database

# -----------------------------------------------------------
# Step 7: FUNCTION → Search Objects
# -----------------------------------------------------------
def search_objects(query_path, database, top_k=5):
    """
    Compare query image embedding with database embeddings using cosine similarity.
    Return top_k most similar objects.
    """
    query_embedding = get_embedding(query_path)
    
    scores = []
    for obj in database:
        similarity = F.cosine_similarity(query_embedding, obj['embedding']).item()
        scores.append({
            'similarity': similarity,
            'object': obj
        })
    
    # sort by similarity (higher = better match)
    scores.sort(key=lambda x: x['similarity'], reverse=True)
    return scores[:top_k]

# -----------------------------------------------------------
# Step 8: FUNCTION → Show Results (Query + Matches)
# -----------------------------------------------------------
def show_results(query_path, results):
    """
    Display the query image and top search results in a grid.
    Also print results in terminal.
    """
    n_results = len(results)
    cols = min(4, n_results + 1)     # max 4 columns
    rows = math.ceil((n_results + 1) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    
    # handle matplotlib's weird axis shape
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    # show query image first
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title("QUERY", fontweight='bold', color='red')
    axes[0].axis('off')
    
    # show each result
    for i, result in enumerate(results, 1):
        obj = result['object']
        axes[i].imshow(obj['cropped_image'])
        axes[i].set_title(f"{obj['source_file']}\n{obj['class']} ({result['similarity']:.3f})")
        axes[i].axis('off')
    
    # hide unused plots
    for i in range(n_results + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # print results in terminal as well
    print("\nResults:")
    for i, result in enumerate(results, 1):
        obj = result['object']
        print(f"{i}. {obj['source_file']} - {obj['class']} (Score: {result['similarity']:.3f})")

# -----------------------------------------------------------
# Step 9: MAIN FUNCTION
# -----------------------------------------------------------
def main():
    """
    Main entry point of the program.
    Builds database (or loads from cache).
    Allows user to select query image.
    Displays search results.
    """
    # make sure query folder exists
    os.makedirs(QUERY_FOLDER, exist_ok=True)
    
    # build/load database
    database = build_database()
    print(f"\nDatabase ready: {len(database)} objects")
    
    while True:
        print(f"\n" + "="*50)
        
        # list query images
        query_files = [f for f in os.listdir(QUERY_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if query_files:
            print("Query images available:")
            for i, f in enumerate(query_files, 1):
                print(f"{i}. {f}")
            print("0. Enter custom path")
            
            choice = input("Select option (or 'exit'): ").strip()
            if choice.lower() == 'exit':
                break
            
            if choice == '0':
                query_path = input("Enter image path: ").strip()
            elif choice.isdigit() and 1 <= int(choice) <= len(query_files):
                query_path = os.path.join(QUERY_FOLDER, query_files[int(choice)-1])
            else:
                print("Invalid choice!")
                continue
        else:
            print("No query images found in 'query_images' folder.")
            query_path = input("Enter query image path (or 'exit'): ").strip()
            if query_path.lower() == 'exit':
                break
        
        if not os.path.exists(query_path):
            print("File not found!")
            continue
        
        print("Searching...")
        results = search_objects(query_path, database)
        
        if results:
            show_results(query_path, results)
        else:
            print("No results found!")

# -----------------------------------------------------------
# Step 10: Run Program
# -----------------------------------------------------------
if __name__ == "__main__":
    main()

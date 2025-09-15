# -----------------------------------------------------------
# HYBRID IMAGE SEARCH (BLIP + CLIP)
# -----------------------------------------------------------
# This script lets you:
# 1. Use BLIP to generate captions for all images
# 2. Use CLIP to create embeddings for both images and captions
# 3. Store results in a cache (so we don’t recompute every time)
# 4. Search images using text queries (hybrid of caption + image similarity)
# 5. Display top matches in a grid with scores + captions
# -----------------------------------------------------------

import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch.nn.functional as F
import pickle
from matplotlib import pyplot as plt
import math

# -----------------------------------------------------------
# Step 1: DEVICE SETUP
# -----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# -----------------------------------------------------------
# Step 2: PATHS AND CACHE FILE
# -----------------------------------------------------------
IMAGE_FOLDER = "images"             # folder where all images are stored
CACHE_FILE = "image_data_cache.pkl" # cache file for storing embeddings & captions

# -----------------------------------------------------------
# Step 3: LOAD MODELS
# -----------------------------------------------------------
print("Loading BLIP model (for captioning)...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=False)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    use_safetensors=True
).to(DEVICE)

print("Loading CLIP model (for embeddings)...")
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# -----------------------------------------------------------
# Step 4: CAPTION GENERATION FUNCTIONS
# -----------------------------------------------------------
def generate_caption(image_path):
    """
    Generate a caption for the image using BLIP.
    Adds safeguards to avoid repetitive or meaningless captions.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output = blip_model.generate(
                **inputs,
                max_length=50,             # limit caption length
                min_length=5,              # ensure at least 5 words
                num_beams=4,               # beam search for better results
                repetition_penalty=1.2,    # discourage repetition
                length_penalty=1.0,
                no_repeat_ngram_size=2,    # avoid repeating 2-word phrases
                do_sample=False            # deterministic output
            )
        
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption if is_valid_caption(caption) else f"Image from {os.path.basename(image_path)}"
    
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return f"Image from {os.path.basename(image_path)}"

def is_valid_caption(caption):
    """
    Check if the generated caption looks valid (not too short, not repetitive).
    """
    if not caption or len(caption.split()) < 3:
        return False
    
    words = caption.lower().split()
    # If too many repeated words
    if len(set(words)) < len(words) * 0.5:
        return False
    
    # If any single word repeats more than 3 times
    if len(words) > 5:
        for word in set(words):
            if words.count(word) > 3:
                return False
    
    return True

# -----------------------------------------------------------
# Step 5: EMBEDDING FUNCTIONS
# -----------------------------------------------------------
def get_image_embedding(image_path):
    """
    Get normalized CLIP embedding for an image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)  # normalize
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_text_embedding(text):
    """
    Get normalized CLIP embedding for text.
    """
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

# -----------------------------------------------------------
# Step 6: LOAD OR BUILD DATABASE
# -----------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print("Loading cached image embeddings...")
    with open(CACHE_FILE, "rb") as f:
        image_data = pickle.load(f)
else:
    print("Building image embeddings from scratch...")
    image_data = []
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    for i, filename in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {filename}")
        img_path = os.path.join(IMAGE_FOLDER, filename)
        
        caption = generate_caption(img_path)
        img_emb = get_image_embedding(img_path)
        
        if img_emb is not None:
            txt_emb = get_text_embedding(caption)
            image_data.append({
                "file": filename,
                "caption": caption,
                "image_embedding": img_emb,
                "text_embedding": txt_emb
            })
    
    # Save cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(image_data, f)
    print(f"Saved embeddings for {len(image_data)} images to {CACHE_FILE}")

# -----------------------------------------------------------
# Step 7: FIX BAD CAPTIONS
# -----------------------------------------------------------
def fix_problematic_captions():
    """
    Re-check captions and regenerate if they are too generic or low-quality.
    """
    global image_data
    
    for i, data in enumerate(image_data):
        if data['caption'].startswith('Image from'):  # fallback caption detected
            print(f"Regenerating caption for {data['file']}...")
            img_path = os.path.join(IMAGE_FOLDER, data['file'])
            
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    output = blip_model.generate(
                        **inputs,
                        max_length=30,
                        min_length=8,
                        num_beams=5,
                        repetition_penalty=1.5,
                        length_penalty=1.2,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        temperature=0.7
                    )
                
                new_caption = blip_processor.decode(output[0], skip_special_tokens=True)
                
                if is_valid_caption(new_caption):
                    print(f"  New caption: {new_caption}")
                    image_data[i]['caption'] = new_caption
                    image_data[i]['text_embedding'] = get_text_embedding(new_caption)
                else:
                    # Fallback: use manual caption
                    manual_caption = f"A person in {data['file'].split('.')[0]}"
                    print(f"  Using manual caption: {manual_caption}")
                    image_data[i]['caption'] = manual_caption
                    image_data[i]['text_embedding'] = get_text_embedding(manual_caption)
            
            except Exception as e:
                print(f"  Error: {e}")
    
    # Save updated cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(image_data, f)
    print("Updated cache saved.")

print("Checking for problematic captions...")
fix_problematic_captions()

# -----------------------------------------------------------
# Step 8: SEARCH FUNCTION
# -----------------------------------------------------------
def search_hybrid(query, top_k=5, caption_weight=0.7):
    """
    Search images using both caption similarity and image similarity.
    - caption_weight: controls importance of caption vs. image (0.0 = only image, 1.0 = only caption).
    """
    query_emb = get_text_embedding(query)
    scores = []
    
    for data in image_data:
        caption_score = F.cosine_similarity(query_emb, data["text_embedding"]).item()
        image_score = F.cosine_similarity(query_emb, data["image_embedding"]).item()
        hybrid_score = caption_weight * caption_score + (1 - caption_weight) * image_score
        
        scores.append({
            "file": data["file"],
            "caption": data["caption"],
            "score": hybrid_score,
            "caption_score": caption_score,
            "image_score": image_score
        })
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]

# -----------------------------------------------------------
# Step 9: DISPLAY RESULTS
# -----------------------------------------------------------
def display_results_grid(results, query):
    """
    Show results in a grid using matplotlib and also print details in terminal.
    """
    if not results:
        print("No matches found.")
        return
    
    n_results = len(results)
    cols = min(3, n_results)
    rows = math.ceil(n_results / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(f'Search Results for: "{query}"', fontsize=16, fontweight='bold')
    
    if n_results == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(results):
        try:
            img_path = os.path.join(IMAGE_FOLDER, result['file'])
            img = Image.open(img_path)
            
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Title with scores
            title = f"{result['file']}\nScore: {result['score']:.3f}"
            title += f" (C: {result['caption_score']:.2f}, I: {result['image_score']:.2f})"
            axes[i].set_title(title, fontsize=10, fontweight='bold')
            
            # Caption below image
            caption = result['caption'][:60] + "..." if len(result['caption']) > 60 else result['caption']
            axes[i].text(0.5, -0.05, caption, transform=axes[i].transAxes,
                        ha='center', va='top', fontsize=9, style='italic')
        
        except Exception as e:
            print(f"Error loading image {result['file']}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\n{result['file']}",
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].axis('off')
    
    for i in range(n_results, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print results in terminal
    print("\nDetailed Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['file']} (Score: {result['score']:.3f})")
        print(f"   Caption: {result['caption']}")
        print(f"   Caption similarity: {result['caption_score']:.3f}, Image similarity: {result['image_score']:.3f}")
        print()

# -----------------------------------------------------------
# Step 10: MAIN LOOP
# -----------------------------------------------------------
def main():
    if not image_data:
        print("No images loaded. Please check your image folder and try again.")
        return
    
    print(f"\nBLIP + CLIP Hybrid Image Search System")
    print(f"Loaded {len(image_data)} images from '{IMAGE_FOLDER}'")
    print("="*60)
    
    while True:
        query = input("\nEnter search query (or 'exit' to quit): ").strip()
        
        if query.lower() in ['exit', 'quit', '']:
            print("Goodbye!")
            break
        
        print(f"\nSearching for: '{query}'")
        
        # number of results
        try:
            top_k = int(input("Number of results (default=5): ").strip() or "5")
            top_k = max(1, min(top_k, len(image_data)))
        except ValueError:
            top_k = 5
        
        # caption weight
        try:
            weight = float(input("Caption weight 0.0-1.0 (default=0.7): ").strip() or "0.7")
            weight = max(0.0, min(weight, 1.0))
        except ValueError:
            weight = 0.7
        
        # perform search
        print(f"\nPerforming hybrid search (caption weight: {weight:.1f})...")
        results = search_hybrid(query, top_k, weight)
        
        # display results
        display_results_grid(results, query)
        
        cont = input("\nSearch again? (y/n, default=y): ").strip().lower()
        if cont in ['n', 'no']:
            print("Goodbye!")
            break

# -----------------------------------------------------------
# Step 11: RUN PROGRAM
# -----------------------------------------------------------
if __name__ == "__main__":
    main()

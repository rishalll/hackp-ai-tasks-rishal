import os
import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
import torch.nn.functional as F
import pickle
import streamlit as st

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = "images"
CACHE_FILE = "image_data_cache.pkl"

st.set_page_config(page_title="BLIP + CLIP Image Search", layout="wide")
st.title("📸 BLIP + CLIP Hybrid Image Search")
st.write("Generate captions with **BLIP** and search images with **CLIP**.")

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        use_safetensors=True
    ).to(DEVICE)

    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    return blip_processor, blip_model, clip_processor, clip_model

blip_processor, blip_model, clip_processor, clip_model = load_models()

# =============================
# FUNCTIONS
# =============================
def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = blip_model.generate(
            **inputs,
            max_length=50, min_length=5,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            do_sample=False
        )
    return blip_processor.decode(output[0], skip_special_tokens=True)

def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

def get_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

@st.cache_data
def build_database():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    database = []
    if not os.path.exists(IMAGE_FOLDER):
        return []

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    for filename in image_files:
        img_path = os.path.join(IMAGE_FOLDER, filename)
        image = Image.open(img_path).convert("RGB")
        caption = generate_caption(image)
        img_emb = get_image_embedding(image)
        txt_emb = get_text_embedding(caption)

        database.append({
            "file": filename,
            "caption": caption,
            "image_embedding": img_emb,
            "text_embedding": txt_emb
        })

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(database, f)
    return database

def search_hybrid(query, database, top_k=5, caption_weight=0.7):
    query_emb = get_text_embedding(query)
    scores = []
    for data in database:
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

# =============================
# UI
# =============================
database = build_database()
if not database:
    st.warning("⚠️ No images found in `images/` folder. Please add some images and restart the app.")

query = st.text_input("🔍 Enter your search query:")
caption_weight = st.slider("⚖️ Caption Weight", 0.0, 1.0, 0.7, 0.1)
top_k = st.slider("📌 Number of Results", 1, 10, 5)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        results = search_hybrid(query, database, top_k, caption_weight)

    st.subheader("Results")
    cols = st.columns(min(3, len(results)))
    for i, result in enumerate(results):
        img_path = os.path.join(IMAGE_FOLDER, result["file"])
        with cols[i % 3]:
            st.image(img_path, caption=f"{result['file']} | Score: {result['score']:.3f}")
            st.caption(result["caption"])

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import pickle
from ultralytics import YOLO
import streamlit as st
import matplotlib.pyplot as plt
import math

# use GPU if available else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = "images"
CACHE_FILE = "objects_cache.pkl"

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    yolo_model = YOLO('yolov8n.pt')
    return clip_model, clip_processor, yolo_model

clip_model, clip_processor, yolo_model = load_models()

# ----------------------------
# Get CLIP embedding
# ----------------------------
def get_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

# ----------------------------
# Detect and crop with YOLO
# ----------------------------
def detect_and_crop_objects(image_path):
    results = yolo_model(image_path, conf=0.3)
    objects = []
    image = Image.open(image_path).convert("RGB")
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_name = yolo_model.names[int(box.cls[0])]
                confidence = box.conf[0].cpu().numpy()
                padding = 10
                x1, y1 = max(0, int(x1-padding)), max(0, int(y1-padding))
                x2, y2 = min(image.width, int(x2+padding)), min(image.height, int(y2+padding))
                cropped = image.crop((x1, y1, x2, y2))
                objects.append({
                    'cropped_image': cropped,
                    'class': class_name,
                    'confidence': float(confidence),
                    'source_file': os.path.basename(image_path)
                })
    return objects

# ----------------------------
# Build DB
# ----------------------------
@st.cache_data
def build_database():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    database = []
    if not os.path.exists(IMAGE_FOLDER):
        return []
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('jpg','png','jpeg'))]
    for filename in image_files:
        img_path = os.path.join(IMAGE_FOLDER, filename)
        objects = detect_and_crop_objects(img_path)
        for obj in objects:
            emb = get_embedding(obj['cropped_image'])
            obj['embedding'] = emb
            database.append(obj)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(database, f)
    return database

database = build_database()

# ----------------------------
# Search
# ----------------------------
def search_objects(query_img, database, top_k=5):
    query_embedding = get_embedding(query_img)
    scores = []
    for obj in database:
        similarity = F.cosine_similarity(query_embedding, obj['embedding']).item()
        scores.append({'similarity': similarity, 'object': obj})
    scores.sort(key=lambda x: x['similarity'], reverse=True)
    return scores[:top_k]

# ----------------------------
# Visualization layouts
# ----------------------------
def show_results(query_img, results, layout="A"):
    n_results = len(results)

    if layout == "A":
        # Vertical layout
        fig, axes = plt.subplots(n_results + 1, 1, figsize=(8, 4*(n_results+1)))
        axes[0].imshow(query_img)
        axes[0].set_title("QUERY", fontweight='bold', color='red')
        axes[0].axis('off')
        for i, result in enumerate(results, 1):
            obj = result['object']
            axes[i].imshow(obj['cropped_image'])
            axes[i].set_title(f"{obj['source_file']} - {obj['class']} ({result['similarity']:.3f})")
            axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

    elif layout == "B":
        # Side-by-side layout
        cols = min(3, n_results)
        rows = math.ceil(n_results / cols)
        fig = plt.figure(figsize=(6*cols, 5*rows))
        gs = fig.add_gridspec(rows, cols+1, width_ratios=[1] + [2]*cols)

        ax_query = fig.add_subplot(gs[:, 0])
        ax_query.imshow(query_img)
        ax_query.set_title("QUERY", fontweight='bold', color='red')
        ax_query.axis('off')

        for i, result in enumerate(results):
            row = i // cols
            col = (i % cols) + 1
            ax = fig.add_subplot(gs[row, col])
            obj = result['object']
            ax.imshow(obj['cropped_image'])
            ax.set_title(f"{obj['source_file']}\n{obj['class']} ({result['similarity']:.3f})", fontsize=11)
            ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔍 YOLO + CLIP Object Search")
st.write("Upload an image and search for similar objects in the dataset.")

uploaded_file = st.file_uploader("Upload query image", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Query Image", use_column_width=True)

    layout_choice = st.radio("Choose layout", ["A - Vertical (large results)", "B - Side-by-side (results bigger than query)"])
    layout = "A" if layout_choice.startswith("A") else "B"

    if st.button("Search"):
        if len(database) == 0:
            st.warning("Database is empty! Please add images in `images/` folder.")
        else:
            results = search_objects(query_img, database)
            if results:
                st.subheader("Top Results:")
                show_results(query_img, results, layout)
            else:
                st.warning("No results found!")

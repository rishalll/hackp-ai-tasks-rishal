# Task 7: Object Search System

An AI-powered object search system that combines **YOLO** (for object detection) and **CLIP** (for semantic understanding) to find similar objects across your image collection.

This is a beginner-friendly project created as part of my project for hackathon in Hackp2025 .

## Demo Video

Watch the project in action: [Task 7 Demo Video](https://youtu.be/HkL7J0d9sxY?si=noYRZbpTWrpIITIA)

## What Does This Do?

This system allows you to:
- **Detect objects** in images automatically using YOLO
- **Search for similar objects** across your entire image collection
- **Use any image as a query** to find visually similar objects
- **View results** with similarity scores in a grid layout
- **Cache processed data** for fast repeated searches

## How It Works

1. **Object Detection**: YOLO finds and crops objects from all your images
2. **Feature Extraction**: CLIP creates numerical representations of each object
3. **Database Creation**: All object features are saved for fast searching
4. **Similarity Search**: Compare query objects with database using cosine similarity
5. **Results Display**: Show the most similar objects with confidence scores

## Features

- **Automatic Object Detection**: Uses YOLOv8 to find objects in images
- **Semantic Understanding**: CLIP embeddings capture visual meaning
- **Smart Caching**: Process images once, search many times
- **Visual Results**: Grid display showing query and matches
- **Similarity Scoring**: Numerical scores showing match confidence
- **Multiple Query Options**: Search using images from query folder or custom paths

## Requirements

### System Requirements
- Python 3.7 or higher
- CUDA-compatible GPU (optional, but recommended for speed)
- At least 4GB RAM (8GB+ recommended)

### Python Libraries
```bash
pip install torch torchvision
pip install transformers
pip install ultralytics
pip install Pillow
pip install matplotlib
pip install pickle-mixin
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. **Download the Code**
```bash
git clone <your-repository-url>
cd task7_object_search
```

### 2. **Prepare Your Images**
- Add your images to the `images/` folder (supports `.jpg`, `.png`, `.jpeg`)
- Add query images to the `query_images/` folder
```
task7_object_search/
├── images/
│   ├── photo1.jpg
│   ├── photo2.png
│   └── photo3.jpeg
├── query_images/
│   ├── query1.jpg
│   └── query2.png
└── task7.py
```

### 3. **Run the Program**
```bash
python task7.py
```

### 4. **First Run Setup**
On the first run, the system will:
- Download YOLO and CLIP models (this may take a few minutes)
- Process all images and detect objects
- Create embeddings for each detected object
- Save everything to `objects_cache.pkl` for faster future searches

### 5. **Start Searching!**
The program will show you available query images, or you can enter a custom path:
```
Query images available:
1. car.jpg
2. dog.png
Select option (or 'exit'): 1
```

## How to Use

### Basic Search Process
1. **Select a query image** from the list or enter a custom path
2. **The system searches** for objects similar to those in your query
3. **Results are displayed** in a grid showing:
   - Your original query image
   - Top 5 most similar objects found
   - Similarity scores for each match
   - Source image filenames

### Understanding Results
Each result shows:
- **Cropped object** detected by YOLO
- **Source filename** where the object was found
- **Object class** (car, person, dog, etc.)
- **Similarity score** (0.0 - 1.0, higher = better match)

### Example Search Session
```
Query images available:
1. car_query.jpg
2. dog_query.png
Select option (or 'exit'): 1

Searching...

Results:
1. sports_car.jpg - car (Score: 0.892)
2. family_car.png - car (Score: 0.834)
3. truck.jpg - truck (Score: 0.721)
4. motorcycle.jpg - motorcycle (Score: 0.645)
5. bicycle.png - bicycle (Score: 0.587)
```

## File Structure

```
task7_object_search/
├── task7.py                  # Main program file
├── strm7.py                  # Helper functions (if used)
├── query_images/             # Put your query images here
│   ├── query1.jpg
│   └── query2.png
├── images/                   # Your image collection to search
│   ├── photo1.jpg
│   └── photo2.png
├── yolov8n.pt               # YOLO model weights (auto-downloaded)
├── requirements.txt          # Python dependencies list
├── objects_cache.pkl         # Auto-generated cache (don't delete!)
└── README.md                 # This file
```

## Advanced Configuration

### YOLO Detection Settings
In the code, you can adjust:
- **Confidence threshold**: `conf=0.3` (higher = fewer, more confident detections)
- **Model size**: `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium)

### Search Parameters
- **Number of results**: Currently returns top 5, can be modified in `search_objects()`
- **Similarity threshold**: Add minimum similarity filtering if needed

### Performance Tips
- **Use GPU**: Automatic if CUDA is available
- **Adjust confidence**: Lower values detect more objects but may include false positives
- **Model choice**: Larger YOLO models are more accurate but slower

## Troubleshooting

### Common Issues

**"No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**"YOLO model not found"**
- The script automatically downloads `yolov8n.pt`
- Check internet connection for first run

**"No objects detected"**
- Try lowering confidence threshold: `conf=0.2`
- Check if images contain recognizable objects
- Verify image quality and format

**"CUDA out of memory"**
- System will automatically fall back to CPU
- Consider using smaller YOLO model (`yolov8n.pt`)
- Process fewer images at once

### Performance Issues

**Slow processing:**
- Ensure GPU is being used (check console output)
- Use smaller YOLO model for faster detection
- Reduce image resolution if very large

**Poor search results:**
- Try different query images
- Check if objects are clearly visible in images
- Verify object classes are supported by YOLO

## Supported Object Classes

YOLO can detect 80 different object classes including:
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: person, dog, cat, horse, cow, sheep
- **Objects**: chair, table, laptop, phone, book
- **Food**: apple, banana, pizza, cake, bottle
- And many more...

## How It Works (Technical Details)

1. **YOLO Detection**: Scans each image and identifies objects with bounding boxes
2. **Object Cropping**: Extracts each detected object as a separate image
3. **CLIP Encoding**: Converts cropped objects into 512-dimensional feature vectors
4. **Database Storage**: Saves all embeddings with metadata (class, confidence, source)
5. **Similarity Search**: Uses cosine similarity to compare query with database
6. **Result Ranking**: Returns most similar objects sorted by similarity score

## About This Project

This project was created as a learning exercise to understand:
- Object detection with YOLO models
- Feature extraction and similarity search
- Computer vision pipelines
- Working with pre-trained AI models
- Building practical search applications

As a student project, the code prioritizes clarity and learning over optimization.

## Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Improve documentation

## License

This project is open source and available for educational purposes. Feel free to use and modify as needed.

## Acknowledgments

- **YOLOv8** by Ultralytics for object detection
- **CLIP** by OpenAI for visual understanding
- **Hugging Face Transformers** for easy model access

---

**Happy Object Searching!**

This is a learning project - if you have questions or suggestions, feel free to reach out!

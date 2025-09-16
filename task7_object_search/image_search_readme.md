# AI-Powered Image Search System

A simple image search system that combines **BLIP** (for image captioning) and **CLIP** (for semantic understanding) to help you find images using natural language queries.

This is a beginner-friendly project created as part of learning AI and computer vision concepts.

## Demo Video

Watch the project in action: [Task 7 Demo Video](https://youtu.be/HkL7J0d9sxY?si=noYRZbpTWrpIITIA)

## What Does This Do?

This system allows you to:
- **Search your image collection** using natural language (e.g., "person wearing red shirt", "sunset over mountains")
- **Automatically generate captions** for all your images
- **Find similar images** based on both visual content and text descriptions
- **View results** in a grid layout with similarity scores

## Features

- **Hybrid Search**: Combines image similarity and caption matching
- **AI-Generated Captions**: Automatically describes your images
- **Smart Caching**: Processes images once, searches instantly thereafter
- **Visual Results**: Shows search results in an organized grid
- **Customizable**: Adjust how much to weight captions vs. images
- **GPU Accelerated**: Uses CUDA if available for faster processing

## Requirements

### System Requirements
- Python 3.7 or higher
- CUDA-compatible GPU (optional, but recommended for speed)
- At least 4GB RAM (8GB+ recommended)

### Python Libraries
```bash
pip install torch torchvision
pip install transformers
pip install Pillow
pip install matplotlib
pip install pickle-mixin
```

Or install all at once:
```bash
pip install torch torchvision transformers Pillow matplotlib pickle-mixin
```

## Quick Start

### 1. **Download the Code**
```bash
git clone <your-repository-url>
cd image-search-system
```

### 2. **Prepare Your Images**
- Create a folder called `images` in the project directory
- Add your image files (supports `.jpg`, `.png`, `.jpeg`)
```
your-project/
├── image_search.py
├── images/
│   ├── photo1.jpg
│   ├── vacation2.png
│   └── family3.jpeg
└── README.md
```

### 3. **Run the Program**
```bash
python task7.py
```

### 4. **First Run Setup**
On the first run, the system will:
- Download AI models (this may take a few minutes)
- Process all your images and generate captions
- Create a cache file (`image_data_cache.pkl`) for faster future searches

### 5. **Start Searching!**
```
Enter search query: person smiling
Number of results (default=5): 3
Caption weight 0.0-1.0 (default=0.7): 0.8
```

## How to Use

### Basic Search
1. **Enter your query**: Describe what you're looking for in natural language
   - "dog running in park"
   - "sunset over water"  
   - "person wearing glasses"
   - "red car"

2. **Choose number of results**: How many matching images to show (default: 5)

3. **Set caption weight**: 
   - `1.0` = Search only based on generated captions
   - `0.5` = Balance between captions and visual similarity  
   - `0.0` = Search only based on visual image content

### Understanding Results
Each result shows:
- **Image thumbnail**
- **Filename** 
- **Generated caption**
- **Total similarity score** (0.0 - 1.0, higher = better match)
- **Caption score** (C: how well query matches the caption)
- **Image score** (I: how well query matches the visual content)

### Example Search Session
```
Enter search query: cat sleeping
Number of results (default=5): 4
Caption weight 0.0-1.0 (default=0.7): 0.6

Results:
1. cat_nap.jpg (Score: 0.847)
   Caption: a cat sleeping on a couch
   Caption similarity: 0.892, Image similarity: 0.781

2. sleepy_pet.png (Score: 0.734)  
   Caption: a fluffy cat lying down peacefully
   Caption similarity: 0.756, Image similarity: 0.698
```

## File Structure

```
task7_object_search/
├── task7.py                  # Main program file
├── strm7.py                  # Helper functions
├── query_images/             # Query images folder
├── images/                   # Your image collection
│   ├── photo1.jpg
│   └── photo2.png
├── yolov8n.pt               # YOLO model weights
├── requirements.txt          # Python dependencies list
├── Hackptasks               # Task instructions
├── image_data_cache.pkl      # Auto-generated cache (don't delete!)
└── README.md                 # This file
```

## Advanced Configuration

### Adjusting Search Behavior

**Caption Weight Settings:**
- `0.0` - Pure visual search (ignores captions)
- `0.3` - Mostly visual, some caption influence  
- `0.7` - Balanced approach (recommended)
- `1.0` - Pure text search (only uses captions)

**When to Use Different Weights:**
- **High caption weight (0.8-1.0)**: When searching for concepts/objects
- **Balanced weight (0.5-0.7)**: General purpose searching  
- **Low caption weight (0.0-0.4)**: When searching for visual style/composition

### Regenerating Cache
If you add new images or want to regenerate captions:
1. Delete `image_data_cache.pkl`
2. Run the program again

## Troubleshooting

### Common Issues

**"No module named 'transformers'"**
```bash
pip install transformers
```

**"CUDA out of memory"**
- The system will automatically use CPU if GPU memory is insufficient
- Consider processing fewer images at once
- Close other GPU-intensive applications

**"No images found"**
- Make sure your images are in the `images/` folder
- Check that image files have supported extensions (`.jpg`, `.png`, `.jpeg`)
- Verify image files aren't corrupted

**Poor search results**
- Try adjusting the caption weight
- Use more specific search terms
- Check that your query matches the visual content or generated captions

### Performance Tips

**Speed up processing:**
- Use GPU if available (automatic)
- Process images in smaller batches if memory is limited
- Keep cache file to avoid reprocessing

**Improve search quality:**
- Use descriptive, specific queries
- Experiment with different caption weights
- Ensure your image collection is well-organized

## How It Works (Technical Overview)

1. **BLIP Model**: Generates natural language captions for each image
2. **CLIP Model**: Creates numerical representations (embeddings) for both images and text
3. **Similarity Matching**: Compares your search query against both image content and captions
4. **Hybrid Scoring**: Combines visual and textual similarity based on your weight settings
5. **Results Ranking**: Shows the most similar images first

## Example Queries That Work Well

- **Objects**: "red car", "black dog", "wooden chair"
- **People**: "person smiling", "child playing", "woman with glasses"  
- **Scenes**: "beach sunset", "mountain landscape", "city street"
- **Actions**: "running in park", "cooking food", "reading book"
- **Descriptions**: "colorful flowers", "old building", "snowy trees"

## About This Project

This project was created as a learning exercise to understand:
- How AI models like BLIP and CLIP work together
- Image processing and computer vision concepts
- Building practical applications with machine learning
- Working with pre-trained models from Hugging Face

As a beginner project, the code focuses on readability and learning rather than optimization.

## Contributing

Feel free to:
- Report bugs or issues
- Suggest new features
- Improve the code
- Add better documentation

## License

This project is open source and available for educational purposes. Feel free to use and modify as needed.

## Acknowledgments

- **BLIP** by Salesforce Research for image captioning
- **CLIP** by OpenAI for multimodal understanding  
- **Hugging Face Transformers** for easy model access

---

**Happy Searching!**

This is a beginner-friendly project - if you have questions or suggestions, feel free to reach out!
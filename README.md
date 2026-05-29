# Paper Bag Detection & Training Pipeline 🛍️

This repository contains a complete pipeline for training a custom object detection model to detect and draw bounding boxes around paper bags. It includes optimized image preprocessing, a local web-based interactive labeling app, and a GPU-accelerated YOLO training script.

---

## 📂 Project Structure

```
paper_bag_dataset/
├── raw_heic/               # Original HEIC images from Apple device (Git-ignored)
├── images/                 # Converted, resized JPEGs (1024px max-dimension)
├── labels/                 # YOLO format text files (class_id x_center y_center width height)
├── templates/
│   └── index.html          # Interactive labelling web app interface
├── convert_and_analyze.py  # Script to batch convert HEIC to JPEG & collect stats
├── label_app.py            # Local Flask server for the labelling interface
├── train_yolo.py           # Shuffles data, splits train/val, and trains YOLOv8
├── venv/                   # Python virtual environment (Git-ignored)
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

To run this pipeline locally, you need Python 3 installed. Clone the repository and initialize a virtual environment:

```bash
# Clone the repository
git clone https://github.com/cronenberg64/train-paper-bag.git
cd train-paper-bag

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install flask ultralytics pillow pyyaml opencv-python
```

### 2. Image Preprocessing (HEIC to JPEG)

To convert the raw HEIC photos to standard, web-friendly JPEG files while sizing them down to an optimized resolution (max 1024px) for labeling and training speed:

1. Place your raw HEIC photos in the `raw_heic/` folder.
2. Run the preprocessing script:
   ```bash
   python3 convert_and_analyze.py
   ```
   *This uses macOS `sips` internally to process images rapidly without heavy external Python dependencies.*

---

## 🏷️ Bounding Box Labelling App

We provide a beautiful local web application to draw bounding boxes and save them in YOLO format.

### Starting the App

```bash
# Ensure venv is active
source venv/bin/activate

# Start Flask local server
python label_app.py
```

Open your browser and navigate to: **`http://127.0.0.1:5001`**

### Labelling Controls
- **Draw Bounding Boxes**: Click and drag your mouse over a paper bag to draw a bounding box.
- **Adjust BBoxes**: Click on a box to select it. Drag corner handles to resize, or drag from the center to reposition it.
- **Search & Filters**: Use the sidebar to search filenames or filter images by **All**, **Unlabeled**, and **Labeled** status.

### ⌨️ Keyboard Shortcuts
| Keybind | Action |
| :--- | :--- |
| **`S`** or **`Enter`** | Save/Commit labels for the current image |
| **`D`** or **`→`** (Right Arrow) | Navigate to next image (with auto-save check) |
| **`A`** or **`←`** (Left Arrow) | Navigate to previous image (with auto-save check) |
| **`Delete`** or **`Backspace`** | Delete selected bounding box |
| **`C`** | Clear all bounding boxes from the current canvas |

---

## 🏋️ Training the YOLOv8 Detector

Once labeling is complete, you can train a YOLO model using the custom dataset:

```bash
# Ensure venv is active
source venv/bin/activate

# Run the training script
python train_yolo.py
```

### What this script does:
1. **Pre-processing split**: Scans labeled images and splits them into an 80/20 train/validation ratio inside the `yolo_dataset/` directory.
2. **YAML generation**: Creates the configuration file `dataset.yaml` with path definitions.
3. **MPS Acceleration**: Automatically checks for Apple Silicon GPU (`MPS`) capabilities to run accelerated training on macOS (falling back to CPU or CUDA depending on environment).
4. **Training execution**: Downloads pretrained weights (`yolov8n.pt`) and trains the model for 30 epochs.
5. **Inference verification**: Runs predictions on 3 random validation images and saves them in the `test_predictions/` folder so you can visually verify detection accuracy.

---

## 📈 Model Output
- Trained weights are saved in: `paper_bag_runs/detect_bags/weights/best.pt`
- Visual predictions on test set are saved in: `test_predictions/`

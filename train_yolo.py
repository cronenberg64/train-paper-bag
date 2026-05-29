import os
import shutil
import random
import yaml
from pathlib import Path

def setup_yolo_dataset(base_dir, train_ratio=0.8):
    """
    Splits the labeled images into train/val sets and creates the YOLO directory structure.
    """
    images_dir = Path(base_dir) / "images"
    labels_dir = Path(base_dir) / "labels"
    yolo_dir = Path(base_dir) / "yolo_dataset"

    # Find all JPEGs that have a label file (even if empty)
    labeled_images = []
    for img_path in images_dir.glob("*.jpg"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            labeled_images.append(img_path)

    if not labeled_images:
        print("Error: No labeled images found. Please label some images first!")
        return None

    print(f"Found {len(labeled_images)} labeled images.")

    # Shuffle and split
    random.seed(42)
    random.shuffle(labeled_images)
    
    split_idx = int(len(labeled_images) * train_ratio)
    train_imgs = labeled_images[:split_idx]
    val_imgs = labeled_images[split_idx:]

    print(f"Splitting into {len(train_imgs)} train and {len(val_imgs)} validation images.")

    # Clean and create directory structure
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    for split in ["train", "val"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_split_files(img_list, split):
        for img_path in img_list:
            # Copy image
            shutil.copy(img_path, yolo_dir / "images" / split / img_path.name)
            # Copy label
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            shutil.copy(lbl_path, yolo_dir / "labels" / split / lbl_path.name)

    copy_split_files(train_imgs, "train")
    copy_split_files(val_imgs, "val")

    # Create dataset.yaml
    dataset_yaml = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "paper_bag"
        }
    }

    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f)

    print(f"YOLO dataset setup complete at: {yolo_dir}")
    print(f"Dataset YAML config generated at: {yaml_path}")
    return yaml_path

def train_model(yaml_path, epochs=50):
    # Import inside function to avoid loading heavy pytorch package if dataset setup fails
    import torch
    from ultralytics import YOLO

    # Check for Apple Silicon GPU acceleration (MPS)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    print(f"Using training device: {device.upper() if isinstance(device, str) else 'CUDA GPU'}")

    # Load YOLOv8 Nano model (pretrained on COCO, excellent starting point)
    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    # Train model
    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        device=device,
        project="paper_bag_runs",
        name="detect_bags"
    )

    print("Training finished!")
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights saved to: {best_weights.resolve()}")
    return model, best_weights

def run_test_inference(model, base_dir):
    """
    Runs prediction on a few validation images to visually verify output.
    """
    yolo_dir = Path(base_dir) / "yolo_dataset"
    val_images_dir = yolo_dir / "images" / "val"
    val_images = list(val_images_dir.glob("*.jpg"))
    
    if not val_images:
        print("No validation images found for testing.")
        return

    test_samples = random.sample(val_images, min(3, len(val_images)))
    print(f"\nRunning test inference on {len(test_samples)} validation sample(s)...")

    results_dir = Path(base_dir) / "test_predictions"
    results_dir.mkdir(exist_ok=True)

    for img_path in test_samples:
        results = model.predict(source=str(img_path), save=False, imgsz=640)
        
        # Save output prediction visualization using plot()
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            
            # Save the image using OpenCV or PIL
            try:
                import cv2
                output_path = results_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(output_path), im_array)
                print(f"Visualization saved to: {output_path}")
            except Exception as e:
                # PIL Fallback
                from PIL import Image
                output_path = results_dir / f"pred_{img_path.name}"
                im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
                im.save(output_path)
                print(f"Visualization saved to: {output_path} (PIL fallback)")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    yaml_config = setup_yolo_dataset(current_dir)
    if yaml_config:
        trained_model, best_pt_path = train_model(yaml_config, epochs=30)
        run_test_inference(trained_model, current_dir)

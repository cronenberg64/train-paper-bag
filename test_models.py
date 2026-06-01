import os
import random
from pathlib import Path
import cv2
from ultralytics import YOLO

def test_on_samples(model_path, model_name, sample_images, output_base):
    print(f"\n--- Testing Model: {model_name} ---")
    model = YOLO(model_path)
    
    model_output_dir = output_base / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample_images:
        results = model.predict(source=str(img_path), imgsz=1024, conf=0.25)
        for r in results:
            im_array = r.plot()
            output_path = model_output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), im_array)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    base_dir = Path("/home/ri-one/train-paper-bag")
    val_images_dir = base_dir / "yolo_dataset/images/val"
    output_dir = base_dir / "test_results_comparison"
    
    # Models to test
    # detect_bags (yolov8n.pt) and detect_bags-2 (yolo26s.pt)
    models = {
        "yolov8n_nano": base_dir / "runs/detect/paper_bag_runs/detect_bags/weights/best.pt",
        "yolo26s_small": base_dir / "runs/detect/paper_bag_runs/detect_bags-2/weights/best.pt"
    }

    # Get sample images
    all_val_images = list(val_images_dir.glob("*.jpg"))
    if not all_val_images:
        # Fallback to general images dir if dataset isn't split
        all_val_images = list((base_dir / "images").glob("*.jpg"))
    
    if not all_val_images:
        print("No images found to test on!")
        exit()

    # Choose up to 5 random images
    num_samples = min(5, len(all_val_images))
    sample_images = random.sample(all_val_images, num_samples)
    print(f"Testing on {len(sample_images)} samples...")

    for name, path in models.items():
        if path.exists():
            test_on_samples(path, name, sample_images, output_dir)
        else:
            print(f"Model not found: {path} (Expected here based on cleanup)")

    print(f"\nDone! Check results in: {output_dir}")

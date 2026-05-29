import os
import subprocess
import time
from pathlib import Path

def convert_heic_to_jpg(input_dir, output_dir, max_dim=1024):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    heic_files = list(input_path.glob("*.HEIC")) + list(input_path.glob("*.heic"))
    if not heic_files:
        print("No HEIC files found in the directory.")
        return

    print(f"Found {len(heic_files)} HEIC files to convert and resize.")
    
    start_time = time.time()
    converted_count = 0
    failed_count = 0

    for idx, filepath in enumerate(heic_files, 1):
        filename = filepath.stem
        output_filepath = output_path / f"{filename}.jpg"

        # Check if already converted
        if output_filepath.exists():
            # If it exists, we can skip or overwrite. We skip to avoid re-processing.
            print(f"[{idx}/{len(heic_files)}] Skipping {filepath.name} (already converted).")
            converted_count += 1
            continue

        print(f"[{idx}/{len(heic_files)}] Converting and resizing {filepath.name}...")
        
        # Build sips command
        # -s format jpeg: set output format to jpeg
        # -Z 1024: resize so maximum dimension is 1024 (maintains aspect ratio)
        cmd = [
            "sips",
            "-s", "format", "jpeg",
            "-Z", str(max_dim),
            str(filepath),
            "--out", str(output_filepath)
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                converted_count += 1
            else:
                print(f"Error converting {filepath.name}: {result.stderr}")
                failed_count += 1
        except Exception as e:
            print(f"Exception converting {filepath.name}: {e}")
            failed_count += 1

    elapsed = time.time() - start_time
    print("\n=== Conversion Summary ===")
    print(f"Total processed: {len(heic_files)}")
    print(f"Successfully converted/verified: {converted_count}")
    print(f"Failed: {failed_count}")
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    heic_dir = os.path.join(current_dir, "raw_heic")
    images_dir = os.path.join(current_dir, "images")
    convert_heic_to_jpg(heic_dir, images_dir)

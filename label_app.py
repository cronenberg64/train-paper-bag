import os
import json
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__, template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
LABELS_DIR = os.path.join(BASE_DIR, 'labels')

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Serve images directly
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

def load_yolo_labels(filename):
    """
    Load YOLO format labels from a text file.
    YOLO Format: <class_id> <x_center> <y_center> <width> <height>
    """
    label_path = os.path.join(LABELS_DIR, f"{os.path.splitext(filename)[0]}.txt")
    boxes = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append({
                            'class_id': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
    return boxes

def save_yolo_labels(filename, boxes):
    """
    Save boxes in YOLO format.
    """
    label_path = os.path.join(LABELS_DIR, f"{os.path.splitext(filename)[0]}.txt")
    try:
        if not boxes:
            # If no boxes, we write an empty file to mark it as processed
            with open(label_path, 'w') as f:
                pass
            return True
            
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n")
        return True
    except Exception as e:
        print(f"Error saving label file {label_path}: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images', methods=['GET'])
def get_images():
    # Scan images dir for JPG/PNG files
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = []
    try:
        for f in sorted(os.listdir(IMAGES_DIR)):
            if f.lower().endswith(valid_exts):
                label_path = os.path.join(LABELS_DIR, f"{os.path.splitext(f)[0]}.txt")
                labeled = os.path.exists(label_path)
                images.append({
                    'filename': f,
                    'labeled': labeled
                })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify(images)

@app.route('/api/image/<filename>', methods=['GET'])
def get_image_details(filename):
    boxes = load_yolo_labels(filename)
    return jsonify({
        'filename': filename,
        'boxes': boxes
    })

@app.route('/api/save', methods=['POST'])
def save_labels():
    data = request.json
    filename = data.get('filename')
    boxes = data.get('boxes', [])
    if not filename:
        return jsonify({'error': 'Filename missing'}), 400
    
    success = save_yolo_labels(filename, boxes)
    if success:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'error': 'Failed to save'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    valid_exts = ('.jpg', '.jpeg', '.png')
    total = 0
    labeled = 0
    try:
        for f in os.listdir(IMAGES_DIR):
            if f.lower().endswith(valid_exts):
                total += 1
                label_path = os.path.join(LABELS_DIR, f"{os.path.splitext(f)[0]}.txt")
                if os.path.exists(label_path):
                    labeled += 1
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({
        'total': total,
        'labeled': labeled,
        'unlabeled': total - labeled
    })

if __name__ == '__main__':
    # Running on local host
    app.run(host='127.0.0.1', port=5001, debug=True)

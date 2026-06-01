import cv2
from ultralytics import YOLO

def run_live_test(model_path, source=0):
    # Load the model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Start webcam capture
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
         # Try source 1 if 0 fails
        print("Source 0 failed, trying source 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open webcam source 0 or 1.")
            return

    print("Starting live detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run inference on the frame
        # stream=True is more memory efficient for real-time
        results = model(frame, stream=True, conf=0.4)

        for r in results:
            annotated_frame = r.plot()
            
            # Display the frame
            cv2.imshow("YOLO Live Detection - Paper Bags", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Using the yolo26s model (detect_bags-2 after cleanup)
    MODEL_PATH = "/home/ri-one/train-paper-bag/runs/detect/paper_bag_runs/detect_bags-2/weights/best.pt"
    
    # Run live test
    run_live_test(MODEL_PATH, source=0)

import cv2
import numpy as np
import onnxruntime as ort
import os
import sys

# Import the ACTUAL SessionDBManager from the backend folder
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)
from session_db import SessionDBManager

YOLO_ONNX = "frontend/models/yolo-face.onnx"
FACENET_ONNX = "frontend/models/edgeface_xs_gamma_06.onnx"

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou

def preprocess_yolo(img, target_size=640):
    img_h, img_w = img.shape[:2]
    
    scale = min(target_size / img_w, target_size / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    pad_x = (target_size - new_w) / 2
    pad_y = (target_size - new_h) / 2
    
    resized = cv2.resize(img, (new_w, new_h))
    padded = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    
    pad_x_int, pad_y_int = int(pad_x), int(pad_y)
    padded[pad_y_int:pad_y_int+new_h, pad_x_int:pad_x_int+new_w] = resized
    
    # YOLO needs RGB, NCHW, normalized to 0-1
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor, scale, pad_x, pad_y

def run_image_test(test_folder="test"):
    print("Loading models...")
    yolo = ort.InferenceSession(YOLO_ONNX, providers=['CPUExecutionProvider'])
    facenet = ort.InferenceSession(FACENET_ONNX, providers=['CPUExecutionProvider'])
    
    if not os.path.exists(test_folder):
        print(f"Error: Could not find test folder {test_folder}")
        return

    print("Initializing ACTUAL backend SessionDBManager...")
    # Using a test directory for sessions
    test_session_dir = os.path.join(test_folder, "sessions_db")
    db = SessionDBManager(base_dir=test_session_dir)
    session_id = db.create_new_session()

    print(f"Starting image processing in folder: {test_folder}")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(valid_exts)]
    
    if not image_files:
        print(f"No images found in {test_folder}.")
        return

    for img_file in image_files:
        img_path = os.path.join(test_folder, img_file)
        print(f"\nProcessing {img_file}...")
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read {img_path}")
            continue
            
        img_h, img_w = frame.shape[:2]
        display_frame = frame.copy()
        
        # 1. Predict faces with YOLO
        tensor, scale, pad_x, pad_y = preprocess_yolo(frame)
        outputs = yolo.run(None, {yolo.get_inputs()[0].name: tensor})[0]
        
        outputs = np.squeeze(outputs)
        outputs = outputs.T
        
        boxes = []
        for out in outputs:
            cx, cy, w, h, conf = out[0], out[1], out[2], out[3], out[4]
            if conf > 0.55:
                # letterbox correction
                x1 = ((cx - w/2) - pad_x) / scale
                y1 = ((cy - h/2) - pad_y) / scale
                x2 = ((cx + w/2) - pad_x) / scale
                y2 = ((cy + h/2) - pad_y) / scale
                boxes.append([x1, y1, x2, y2, conf])
                
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        kept_boxes = []
        suppressed = set()
        for i in range(len(boxes)):
            if i in suppressed: continue
            kept_boxes.append(boxes[i])
            for j in range(i+1, len(boxes)):
                if j in suppressed: continue
                if compute_iou(boxes[i][:4], boxes[j][:4]) > 0.4:
                    suppressed.add(j)

        # 2. Extract embeddings and query ACTUAL backend DB
        for box in kept_boxes:
            x1, y1, x2, y2, conf = box
            
            # Use identical logic from frontend app.js
            bw, bh = x2 - x1, y2 - y1
            side = max(bw, bh) * 1.1
            shiftY = side * 0.05
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            c_x1 = int(max(0, cx - side/2))
            c_y1 = int(max(0, (cy - shiftY) - side/2))
            c_x2 = int(min(img_w, cx + side/2))
            c_y2 = int(min(img_h, (cy - shiftY) + side/2))
            
            if c_x2 - c_x1 < 20 or c_y2 - c_y1 < 20: 
                continue
                
            crop = frame[c_y1:c_y2, c_x1:c_x2]
            crop_h, crop_w = crop.shape[:2]
            
            t_size = 112
            c_scale = min(t_size/crop_w, t_size/crop_h)
            n_w, n_h = int(crop_w * c_scale), int(crop_h * c_scale)
            p_x, p_y = (t_size - n_w)//2, (t_size - n_h)//2
            
            resized_crop = cv2.resize(crop, (n_w, n_h))
            final_crop = np.zeros((t_size, t_size, 3), dtype=np.uint8)
            final_crop[p_y:p_y+n_h, p_x:p_x+n_w] = resized_crop
            
            # InsightFace/EdgeFace RGB and normalized to (v - 127.5) / 127.5
            rgb_crop = cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)
            final_tensor = np.zeros((3, 112, 112), dtype=np.float32)
            final_tensor[0, :, :] = (rgb_crop[:, :, 0] - 127.5) / 127.5
            final_tensor[1, :, :] = (rgb_crop[:, :, 1] - 127.5) / 127.5
            final_tensor[2, :, :] = (rgb_crop[:, :, 2] - 127.5) / 127.5
            final_tensor = np.expand_dims(final_tensor, axis=0)
            
            # Embed
            res = facenet.run(None, {facenet.get_inputs()[0].name: final_tensor})[0][0]
            
            # L2 Normalize
            norm = np.linalg.norm(res)
            emb = (res / (norm + 1e-10)).tolist() # Session DB expects list or array 
            
            # Query ACTUAL backend SessionDBManager! (0.55 threshold from main.py)
            best_reid, best_name, similarity = db.query_face(emb, threshold=0.55)
            
            if best_reid is None:
                if similarity < 0.45:
                    new_reid = db.add_face(emb, "Unknown")
                    db.update_name(new_reid, f"Person_{new_reid}")
                    best_reid = new_reid
                    best_name = f"Person_{new_reid}"
                    print(f" -> ADDED new face: {best_name} (sim: {similarity:.4f})")
                else:
                    best_name = "Unknown"
                    print(f" -> GREY ZONE face (sim: {similarity:.4f}). Not matched, not added.")
            else:
                print(f" -> MATCHED: {best_name} (sim: {similarity:.4f})")
                
            # Draw
            color = ((hash(best_name) * 50) % 255, (hash(best_name) * 100) % 255, (hash(best_name) * 150) % 255)
            if best_reid is None: color = (0, 0, 255) # Red for unknown
            
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{best_name}" if best_reid else f"Unknown"
            if similarity:
                label += f" (D: {similarity:.2f})"
            cv2.putText(display_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        cv2.imshow("Face Match using SessionDB", display_frame)
        print("Press any key to continue to the next image, or 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    # Cleanup test session
    print(f"Cleaning up test session {session_id}...")
    db.delete_session(session_id)

if __name__ == "__main__":
    test_folder_path = "test"
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
        print(f"Created '{test_folder_path}' folder. Please place test images there and run again.")
        run_image_test(test_folder_path) # Might be empty, will just print "No images"
    else:
        run_image_test(test_folder_path)

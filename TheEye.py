import cv2
import numpy as np
import face_recognition
import json
import os
import logging
import time
from imutils.object_detection import non_max_suppression
import imutils

# --- Configuration and Setup ---

# Create output directories if they don't exist
os.makedirs("detected_objects", exist_ok=True) # This directory will store the saved objects
os.makedirs("screen_captures", exist_ok=True)
os.makedirs("detected_faces", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("combined_recognition.log"),
        logging.StreamHandler()
    ]
)
logging.info("Starting combined recognition script.")

# --- Face Recognition Setup ---
# Load known faces
# Initialize new_face_counter for unique naming of unknown faces
new_face_counter = 1

try:
    obama_image = face_recognition.load_image_file("skrillex1.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    logging.info("Loaded skrillex1.jpg for face recognition.")
except IndexError:
    logging.error("Could not find face in skrillex1.jpg. Ensure the image contains a clear face.")
    obama_face_encoding = None
except FileNotFoundError:
    logging.error("skrillex1.jpg not found. Face recognition for Skrillex will be skipped.")
    obama_face_encoding = None

try:
    biden_image = face_recognition.load_image_file("alvin.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    logging.info("Loaded alvin.jpg for face recognition.")
except IndexError:
    logging.error("Could not find face in alvin.jpg. Ensure the image contains a clear face.")
    biden_face_encoding = None
except FileNotFoundError:
    logging.error("alvin.jpg not found. Face recognition for Alvin will be skipped.")
    biden_face_encoding = None

known_face_encodings = []
known_face_names = []

if obama_face_encoding is not None:
    known_face_encodings.append(obama_face_encoding)
    known_face_names.append("Skrillex")
if biden_face_encoding is not None:
    known_face_encodings.append(biden_face_encoding)
    known_face_names.append("Alvin")

if not known_face_encodings:
    logging.warning("No known faces loaded for face recognition. Face recognition will not function.")
else:
    # Adjust new_face_counter based on initial known faces
    new_face_counter = len(known_face_names) + 1


# --- YOLO Object Detection Setup ---
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    logging.info("YOLO backend set to OpenCL.")
except Exception as e:
    logging.warning(f"Failed to set YOLO backend to OpenCL: {e}. Falling back to CPU.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = []
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    logging.info(f"Loaded {len(classes)} COCO class names.")
except FileNotFoundError:
    logging.error("coco.names not found. Object detection will use numerical class IDs.")
    classes = [str(i) for i in range(80)]  # Default to 80 common classes if names file is missing

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# --- Text Detection Setup ---
# Load the pre-trained EAST text detector
east_net = cv2.dnn.readNet("frozen_east_text_detection.pb")
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# --- Function to Decode Predictions ---
def decode_predictions(scores, geometry, min_confidence=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

# --- Video Capture Setup ---
cap = cv2.VideoCapture("video1.mp4")
if not cap.isOpened():
    logging.error("Error: Could not open video file 'video1.mp4'. Exiting.")
    exit()
logging.info("Video capture initialized from 'video1.mp4'.")

# --- Main Loop Variables ---
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True  # For face recognition frame skipping
object_dict = {}  # Stores unique objects detected across frames
detection_history = []  # Detailed log of all detections
frame_count = 0
appearance_log = {}  # Tracks appearances of objects

# Global counter for saving objects to ensure unique filenames
object_save_counter = 0

# --- Main Processing Loop ---
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1000)
    if not ret:
        logging.info("End of video stream or error reading frame. Exiting loop.")
        break

    frame_count += 1
    current_timestamp = time.time()  # Get current timestamp

    # --- Face Recognition Processing ---
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            name = "Unknown"
            if known_face_encodings:  # Only compare if there are known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    # This is a new, unknown face. Save it.
                    unique_face_id = f"Unknown_Face_{new_face_counter}_F{frame_count}_P{i}"
                    
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(unique_face_id)
                    name = unique_face_id
                    new_face_counter += 1

                    # Save the detected face image
                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    top = max(0, top)
                    right = min(frame.shape[1], right)
                    bottom = min(frame.shape[0], bottom)
                    left = max(0, left)

                    face_image = frame[top:bottom, left:right]
                    face_filename = f"detected_faces/{unique_face_id}.jpg"
                    cv2.imwrite(face_filename, face_image)
                    logging.info(f"Frame {frame_count}: New unknown face detected - {unique_face_id}. Saved as {face_filename}")
            else:
                # No known faces loaded, so all detected faces are "Unknown"
                unique_face_id = f"Unknown_Face_{new_face_counter}_F{frame_count}_P{i}"
                known_face_encodings.append(face_encoding)
                known_face_names.append(unique_face_id)
                name = unique_face_id
                new_face_counter += 1

                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                top = max(0, top)
                right = min(frame.shape[1], right)
                bottom = min(frame.shape[0], bottom)
                left = max(0, left)

                face_image = frame[top:bottom, left:right]
                face_filename = f"detected_faces/{unique_face_id}.jpg"
                cv2.imwrite(face_filename, face_image)
                logging.info(f"Frame {frame_count}: New unknown face detected (no known faces) - {unique_face_id}. Saved as {face_filename}")

            face_names.append(name)

    process_this_frame = not process_this_frame  # Toggle for next frame

    # --- YOLO Object Detection Processing ---
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    current_frame_objects = set()  # To track objects in the current frame

    if isinstance(indices, np.ndarray):  # Ensure indices is an array if detections exist
        for i, idx in enumerate(indices.flatten()):  # Flatten in case of multiple dimensions
            box = boxes[idx]
            x, y, w, h = box

            class_id = class_ids[idx]
            current_frame_objects.add(class_id)

            object_name = classes[class_id] if class_id < len(classes) else f"Class_{class_id}"

            # --- Save Detected Object Image ---
            # Ensure coordinates are within frame bounds for cropping
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            if x2 > x1 and y2 > y1: # Ensure valid crop region
                detected_object_image = frame[y1:y2, x1:x2]
                object_save_counter += 1
                # Unique filename: object_name_frame_count_instance_counter.jpg
                object_filename = f"detected_objects/{object_name}_{frame_count}_{i}_{object_save_counter}.jpg"
                cv2.imwrite(object_filename, detected_object_image)
                logging.info(f"Frame {frame_count}: Saved detected object '{object_name}' as {object_filename}")

            if class_id not in object_dict:
                # New object type detected (first time seeing this class_id)
                object_dict[class_id] = {
                    "name": object_name,
                    "first_detected": current_timestamp
                }
                logging.info(f"Frame {frame_count}: New object type detected - {object_name}")

            # Draw bounding box and label for YOLO detections
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{object_name} {confidences[idx]:.2f}",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detection_history.append({
                "frame": frame_count,
                "timestamp": current_timestamp,
                "object_id": class_id,
                "object_name": object_name,
                "confidence": confidences[idx],
                "box": box
            })

    # --- Text Detection Processing ---
    # Prepare the frame for text detection
    newW, newH = 320, 320
    frame_resized = cv2.resize(frame, (newW, newH))
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward(layerNames)

    # Decode predictions
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Draw text detection boxes and log detected text
    for (startX, startY, endX, endY) in boxes:
        # Scale text detection boxes back to original frame size
        startX = int(startX * (width / newW))
        startY = int(startY * (height / newH))
        endX = int(endX * (width / newW))
        endY = int(endY * (height / newH))

        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        logging.info(f"Frame {frame_count}: Text detected at [{startX}, {startY}, {endX}, {endY}]")

    # --- Draw Face Recognition Results on Frame ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # --- Display Results ---
    cv2.imshow("Combined Recognition", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("User pressed 'q'. Exiting.")
        break

# --- Finalization ---
# Save detection results to JSON
detection_data = {
    "detections": detection_history,
    "objects": object_dict,
    "total_objects": len(object_dict),
    "total_frames": frame_count
}

try:
    with open('detection_results.json', 'w') as f:
        json.dump(detection_data, f, indent=4)
    logging.info("Detection results saved to detection_results.json.")
except Exception as e:
    logging.error(f"Error saving detection_results.json: {e}")

cap.release()
cv2.destroyAllWindows()
logging.info(f"Detection complete. Found {len(object_dict)} unique object types and saved {object_save_counter} object instances.")
logging.info("Script finished.")


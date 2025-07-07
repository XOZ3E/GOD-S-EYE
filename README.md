# GOD'S_EYE
# WHAT_IS_THIS?
A real-time vision system that watches everything 👁️. Inspired by the surveillance tech in **Fast & Furious**, but built in reality.  
This is a **base project** still under development using the **Darknet framework**, designed to recognize objects, faces, and text in video.
# A_REAL-TIME_MULTI-MODAL_RECOGNITION_SYSTEM
# STILL UNDER DEVELOPMENT USING DARKNET FRAMEWORK
# WHAT_IT_DOES
- Real-time video stream processing (from file or webcam)
- Object detection using YOLOv4
  - Crops and saves detected objects by class + frame
- Face recognition using Dlib (via face_recognition lib)
  - Learns unknown faces automatically
- Text detection using EAST
  - Locates text blocks in frames
- Logs everything (JSON + LOG files)
- Organizes outputs into folders
![space_holder](https://github.com/XOZ3E/GOD-S-EYE/blob/main/mark1.png)
# NEURAL_NETWORK_BACKBONE
- YOLOv4 (You Only Look Once)
  - yolov4.cfg: Network architecture
  - yolov4.weights: Learned knowledge
  - coco.names: Object class names
- OpenCV (cv2)
  - Video processing, drawing, saving images, etc.
- face_recognition (based on dlib)
  - 128-dim face encodings
- EAST
  - Efficient and accurate text detection

# SETUP_AND_REQUIREMENTS
# I DIDN'T UPLODE THE WHOLE CODE CONTACT IF YOU WANNA CONTRIBUTE AND USE IT 
Put these files in the same folder as `TheEye.py`:
- `yolov4.cfg`
- `yolov4.weights`
- `coco.names`
- `frozen_east_text_detection.pb`

# OUTPUT_STRUCTURE
- `detected_objects/` - All object crops
- `detected_faces/` - All known + unknown face crops
- `screen_captures/` - (planned for full-frame saves)
- `combined_recognition.log` - Full text log
- `detection_results.json` - Frame/object summary
![space_holder](https://github.com/XOZ3E/GOD-S-EYE/blob/main/main.gif)
# PLANNED_FEATURES
- Faster processing (FPS boost)
- GUI frontend (click instead of code)
- Database integration (SQLite/PostgreSQL)
- Object/face tracking with unique IDs
- Real-time alerts
- Webcam input toggle
- OCR text extraction from detected text

# WATCH HERE
[Youtube](https://www.youtube.com/watch?v=5WGWMRBSylE)

# VIDEO USED FORM PEXELS
[**Video used**](https://www.pexels.com/video/officers-investigating-7253660)

# LICENSE_AND_DISCLAIMER
🚫 NOT FOR COMMERCIAL USE  
This project is **NOT LICENSED** for public or commercial distribution.  
Built for learning and concept testing only.  
Contact maintainers directly to collaborate responsibly.


## CONTACT
All queries and collaborations are welcome.  
For help, suggestions, or ideas:
- **Name**: XOZ3E
- **Telegram**: https://t.me/XCZGITHUB
- **GitHub**: [XOZE😁](https://github.com/XOZ3E)



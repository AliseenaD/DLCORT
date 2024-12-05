import cv2
import os

def crop_video(input_path, output_path, y1, y2, x1, x2):
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (x2 - x1, y2 - y1)
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Crop the frame
        cropped_frame = frame[y1:y2, x1:x2]
        out.write(cropped_frame)
    
    cap.release()
    out.release()

# Example usage:
crop_video("/Users/alidaeihagh/Desktop/507X_1_d1DLC_resnet50_ObjectTestNov21shuffle1_250000_labeled.mp4", "/Users/alidaeihagh/Desktop/507X_1_d1DLC_cropped.mp4", y1=77, y2=342, x1=159, x2=447)
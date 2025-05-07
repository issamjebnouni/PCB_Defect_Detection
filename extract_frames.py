import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Convert BGR (OpenCV default) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Construct filename and save the RGB frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        # Save using RGB order
        cv2.imwrite(frame_filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))  # convert back to BGR for saving
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'.")

# Example usage:
video_path = '20250504_155442.mp4'
output_folder = 'frames_output'
extract_frames(video_path, output_folder)

import cv2
from pathlib import Path

def read_video(video_path):
    cap= cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    frames=[]
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    while True:
        ret, frame =cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def save_video(output_video_frames,output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("No frames were provided to save_video.")
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path),fourcc, fps, (output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
   

    

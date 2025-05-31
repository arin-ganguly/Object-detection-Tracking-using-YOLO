from ultralytics import YOLO
import cv2
import time

# ---------------------- Load Model ----------------------
def load_model(model_path):
    return YOLO(model_path)

# -------------------- Initialize Video ----------------------
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")
    return cap, fps, total_frames

# -------------------- Process Frame ----------------------
def process_frame(frame, model):
    # frame = cv2.resize(frame, (640, 360))  # Resize incase of faster processing
    results = model.predict(source=frame, conf=0.25, verbose=False)
    return results[0].plot()

# ---------------------- Draw Overlay ----------------------
def draw_overlay(frame, fps, current_frame, total_frames):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return frame

# ---------------------- Handle Keyboard Input ----------------------
def handle_keys(key, cap, fps, paused):
    if key == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        return 'quit', paused
    elif key == ord('p'):
        return 'pause', not paused
    elif key == ord('d'):
        current = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current + int(fps * 5))
        print(f"Forwarded to frame {int(current + fps * 5)}")
    elif key == ord('a'):
        current = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(current - int(fps * 5), 0))
        print(f"Rewound to frame {int(max(current - fps * 5, 0))}")
    return None, paused

# -------------------- Main Loop --------------------
def run_video(model_path, video_path):
    model = load_model(model_path)
    cap, fps_video, frame_count = load_video(video_path)
    
    prev_time = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            start_time = time.time()

            processed_frame = process_frame(frame, model)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            output = draw_overlay(processed_frame, fps, current_frame, frame_count)
            cv2.imshow('frame', output)

        key = cv2.waitKey(1) & 0xFF
        action, paused = handle_keys(key, cap, fps_video, paused)
        if action == 'quit':
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------- main function -------------- 
if __name__ == "__main__":
    model_path = '../YOLO weights/yolov10n.pt'
    video_path = r"E:\YOLOv8\Videos\cars.mp4"
    run_video(model_path, video_path)

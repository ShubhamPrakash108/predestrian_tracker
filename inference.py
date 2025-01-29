

from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("/content/drive/MyDrive/YOLO/best_final.pt")

target_id_input = input("Enter the ID to search (press Enter to skip): ")
target_id = int(target_id_input) if target_id_input.strip() else None

input_video_path = "/content/drive/MyDrive/YOLO/2025-01-29 09-18-38.mp4"
output_video_path = "/content/output_video5.mp4"

video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

history = {}
frame_count = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
    else:
        boxes = []
        ids = []

    for box, id in zip(boxes, ids):
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        if id not in history:
            history[id] = []
        history[id].append((cx, cy))

        if len(history[id]) > 20:
            history[id].pop(0)

    for id, points in history.items():
        if len(points) >= 2:
            hue = (id * 50) % 180
            saturation = 255

            for i in range(1, len(points)):
                prev = points[i-1]
                curr = points[i]

                brightness = 255 - int((i / (len(points)-1)) * 200)
                value = max(55, brightness)

                color = cv2.cvtColor(
                    np.uint8([[[hue, saturation, value]]]),
                    cv2.COLOR_HSV2BGR
                )[0][0]
                color = tuple(map(int, color))

                cv2.line(annotated_frame, prev, curr, color, 2)

    if target_id is not None and target_id in ids:
        idx = np.where(ids == target_id)[0][0]
        x1, y1, x2, y2 = map(int, boxes[idx])

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = f"Target ID: {target_id}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    output_video.write(annotated_frame)

video.release()
output_video.release()
print(f"Output video saved at: {output_video_path}")





from IPython.display import HTML
from base64 import b64encode

video_path = "/content/output_video5.mp4"

def show_video(video_path):
    with open(video_path, "rb") as f:
        video_base64 = b64encode(f.read()).decode()
    return HTML(f'<video width="640" height="360" controls><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video>')

show_video(video_path)












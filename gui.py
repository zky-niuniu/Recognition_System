import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# 加载YOLOv8模型
model = YOLO('runs/detect/train5/weights/best.pt')  # 假设已经有一个针对人体动作识别微调过的模型

def process_image(image):
    results = model(image)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            conf = box.conf
            cls = int(box.cls)
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
            cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

st.title("Human Action Recognition System")

option = st.sidebar.selectbox("Choose Input Type", ("Image", "Video", "Camera"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        processed_image = process_image(image)
        st.image(processed_image, caption="Processed Image")
        if st.button('Save Result'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                Image.fromarray(processed_image).save(tmp_file.name)
                st.write(f"Saved to {tmp_file.name}")

elif option == "Camera":
    run = st.checkbox('Open/Close Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = process_image(frame)
        FRAME_WINDOW.image(processed_frame)
        if any([result.boxes.shape[0] > 0 for result in model(frame)]):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                Image.fromarray(processed_frame).save(tmp_file.name)
                st.write(f"Screenshot saved to {tmp_file.name}")
    else:
        camera.release()

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=['mp4'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_image(frame)
            st.video(processed_frame)
            if st.button('Save Video Result'):
                # 这里添加保存处理后视频的逻辑
                pass
        cap.release()
import streamlit as st
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import datetime

SAVE_DIR="/Volumes/Game/Saved_results"
os.makedirs(SAVE_DIR,exist_ok=True)
# 安全加载模型

model = YOLO('/Volumes/Game/rec/gui/best.pt')


def process_image(image):
    """处理图像并绘制检测结果"""
    results = model(image)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            
            # 安全处理置信度（解决numpy数组格式化问题）
            conf = box.conf
            if isinstance(conf, np.ndarray):
                conf = float(conf[0]) if conf.size > 0 else 0.0
            else:
                conf = float(conf)
            
            # 安全处理类别
            cls = box.cls
            if isinstance(cls, np.ndarray):
                cls = int(cls[0]) if cls.size > 0 else 0
            else:
                cls = int(cls)
            
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
            cv2.putText(image, label, (r[0], r[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

st.title("Manhole Cover Recognition System")

option = st.sidebar.selectbox("Choose Input Type", ("Image", "Video", "Camera"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            if len(image.shape) == 2:  # 如果是灰度图
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:   # 如果是RGBA图
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
            processed_image = process_image(image)
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            
            if st.button('Save Result'):
                save_path=os.path.join(SAVE_DIR,f"result_{len(os.listdir(SAVE_DIR))}.jpg")
                Image.fromarray(processed_image).save(save_path)
                st.success(f"Saved to {save_path}")
        except Exception as e:
            st.error(f"图像处理错误: {e}")

elif option == "Camera":
    run = st.checkbox('Open Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    
    if run:
        try:
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.warning("无法从摄像头获取帧")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_image(frame)
                FRAME_WINDOW.image(processed_frame)
                
                # 检测到对象时保存截图
                results = model(frame)
                if any(len(r.boxes) > 0 for r in results):
                    save_dir="/Volumes/Game/results/screenshots"
                    os.makedirs(save_dir,exist_ok=True)
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path=os.path.join(save_dir,f"screenshot_{timestamp}.jpg")
                    Image.fromarray(processed_frame).save(save_path)
                    st.info(f"Detection saved to: {save_path}")
                    st.experimental_rerun()  # 刷新以避免重复保存
        except Exception as e:
            st.error(f"摄像头错误: {e}")
        finally:
            camera.release()
            if run:  # 如果仍然勾选但出错
                run = False
                st.experimental_rerun()

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            save_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_image(frame)
                stframe.image(processed_frame)
                
            cap.release()
            
            if st.button('Save Video Result'):
                # 这里需要添加视频保存逻辑
                st.warning("视频保存功能待实现")
                
        except Exception as e:
            st.error(f"视频处理错误: {e}")
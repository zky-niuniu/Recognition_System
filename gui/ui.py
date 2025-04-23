import streamlit as st
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import datetime
import time

# 配置区域 ================================================
SAVE_DIR = "/Volumes/Game/Saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 初始化Session State
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_save_time' not in st.session_state:
    st.session_state.last_save_time = 0

# 侧边栏配置 =============================================
st.sidebar.title("Settings")
MODEL_PATH = st.sidebar.text_input("Model Path", "/Volumes/Game/rec/gui/best.pt")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
auto_save = st.sidebar.checkbox("Enable Auto-Save", True)
save_cooldown = st.sidebar.number_input("Save Cooldown (seconds)", 1, 60, 5)

# 加载模型
try:
    model = YOLO(MODEL_PATH)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# 图像处理函数 ===========================================
def process_image(image, conf_threshold=0.6):
    results = model(image)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box.conf, '__iter__') else float(box.conf)
            
            if conf >= conf_threshold:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0]) if hasattr(box.cls, '__iter__') else int(box.cls)
                label = f'{model.names[cls]} {conf:.2f}'
                
                cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)
                cv2.putText(image, label, (r[0], r[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image, results

# 主界面 ================================================
st.title("Manhole Cover Recognition System")
option = st.radio("Input Type", ("Image", "Video", "Camera"), horizontal=True)

# 图像处理分支
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            try:
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Original Image", use_container_width=True)
            except Exception as e:
                st.error(f"图像加载错误: {e}")
        
        with col2:
            try:
                image_np = np.array(original_image)
                if len(image_np.shape) == 2:  # 灰度图
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                elif image_np.shape[2] == 4:   # RGBA图
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                
                processed_image, results = process_image(image_np, conf_threshold)
                st.image(processed_image, caption="Processed Image", use_container_width=True)
                
                if st.button('Save Result'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(SAVE_DIR, f"result_{timestamp}.jpg")
                    Image.fromarray(processed_image).save(save_path)
                    st.toast(f"Saved to {save_path}", icon="✅")
                    
                    # 显示检测结果
                    if results and len(results[0].boxes) > 0:
                        st.json({
                            "detections": len(results[0].boxes),
                            "classes": [model.names[int(box.cls)] for box in results[0].boxes],
                            "confidences": [float(box.conf) for box in results[0].boxes]
                        })
            except Exception as e:
                st.error(f"图像处理错误: {e}")

# 视频处理分支
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        
        col1, col2 = st.columns(2)
        with col1:
            st.video(uploaded_video)
        
        with col2:
            if st.button("Process Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    cap = cv2.VideoCapture(tfile.name)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    output_path = os.path.join(SAVE_DIR, f"processed_{uploaded_video.name}")
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                    
                    frame_placeholder = st.empty()
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame, _ = process_image(frame_rgb, conf_threshold)
                        out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                        
                        # 显示处理进度
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        progress = min(current_frame / total_frames, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {current_frame}/{total_frames} frames")
                        
                        # 显示当前帧（每10帧更新一次以提高性能）
                        if current_frame % 10 == 0:
                            frame_placeholder.image(processed_frame)
                    
                    out.release()
                    st.success(f"Processing complete! Saved to {output_path}")
                    st.video(output_path)
                
                except Exception as e:
                    st.error(f"视频处理错误: {e}")
                finally:
                    cap.release() if 'cap' in locals() else None
                    out.release() if 'out' in locals() else None
                    os.unlink(tfile.name)

# 摄像头处理分支
elif option == "Camera":
    st.warning("Camera may take a few seconds to initialize")
    
    if st.button("Start Camera", disabled=st.session_state.camera_active):
        st.session_state.camera_active = True
    
    if st.button("Stop Camera", disabled=not st.session_state.camera_active):
        st.session_state.camera_active = False
        st.experimental_rerun()
    
    if st.session_state.camera_active:
        FRAME_WINDOW = st.empty()
        camera = cv2.VideoCapture(0)
        
        try:
            while st.session_state.camera_active:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture frame from camera")
                    st.session_state.camera_active = False
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame, results = process_image(frame_rgb, conf_threshold)
                FRAME_WINDOW.image(processed_frame)
                
                # 自动保存逻辑
                if auto_save and results and len(results[0].boxes) > 0:
                    current_time = time.time()
                    if current_time - st.session_state.last_save_time > save_cooldown:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(SAVE_DIR, f"camera_{timestamp}.jpg")
                        cv2.imwrite(save_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                        st.session_state.last_save_time = current_time
                        st.sidebar.success(f"Detection saved: {save_path}")
        
        except Exception as e:
            st.error(f"Camera error: {e}")
            st.session_state.camera_active = False
        finally:
            camera.release()
            if st.session_state.camera_active:
                st.session_state.camera_active = False
                st.experimental_rerun()

# 添加页脚
st.markdown("---")
st.caption("Manhole Cover Detection System | © 2023")
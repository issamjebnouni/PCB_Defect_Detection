import os
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
import supervision as sv
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ────────────────────────────────────────────────────────────────────────────────
# 1. Environment Setup & Optimizations
# ────────────────────────────────────────────────────────────────────────────────
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_WATCHER_BLACKLIST"] = os.path.dirname(torch.__file__)
class DummyPath: _path = []
torch.classes.__path__ = DummyPath()

# ────────────────────────────────────────────────────────────────────────────────
# 2. Streamlit Page Configuration
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 YOLOv12 Real-Time Detection")
st.sidebar.header("Configuration")

# ────────────────────────────────────────────────────────────────────────────────
# 3. Model Loading with GPU Optimization
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("defect_detector.pt").to(device)
    # Warm-up inference with correct input shape
    dummy = torch.zeros((1, 3, 640, 640), device=device)
    model.predict(dummy, conf=0.25, verbose=False)
    return model

model = load_model()
st.sidebar.success(f"✅ Model loaded (Using {'GPU' if torch.cuda.is_available() else 'CPU'})")

# ────────────────────────────────────────────────────────────────────────────────
# 4. Annotation Tools Setup
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data
def setup_annotators():
    return (
        sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=2
        ),
        sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
            text_scale=0.2,
            text_thickness=1
        )
    )

bbox_annotator, label_annotator = setup_annotators()

# ────────────────────────────────────────────────────────────────────────────────
# 5. Video Processing Class (WebRTC)
# ────────────────────────────────────────────────────────────────────────────────
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence_threshold = st.session_state.get("conf_threshold", 0.25)
        
    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        labels = [
            f"{self.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in
            zip(detections.class_id, detections.confidence)
        ]
        
        annotated_frame = bbox_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        return annotated_frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        
        # Optional frame resizing for performance
        if st.session_state.get("downscale", True):
            image = cv2.resize(image, (640, 480))
        
        processed_image = self.annotate_frame(image)
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

# ────────────────────────────────────────────────────────────────────────────────
# 6. User Interface Components
# ────────────────────────────────────────────────────────────────────────────────
def image_upload_section():
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    if uploaded_file is not None:
        image = cv2.imdecode(
            np.frombuffer(uploaded_file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        
        with st.spinner("Processing image..."):
            annotated_image = VideoProcessor().annotate_frame(image)
            st.image(
                cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Processed Image"
            )

# ────────────────────────────────────────────────────────────────────────────────
# 7. Main Application Flow
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # Sidebar controls
    st.sidebar.subheader("Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        key="conf_threshold"
    )
    
    downscale = st.sidebar.toggle(
        "Enable Downscaling (Better Performance)",
        value=True,
        key="downscale"
    )
    
    # Mode selection
    app_mode = st.sidebar.radio(
        "Select Mode:",
        ["Live Camera", "Image Upload"],
        index=0
    )
    
    if app_mode == "Image Upload":
        image_upload_section()
    else:
        st.subheader("Live Camera Detection")
        st.write("Click 'Start' below to begin real-time detection")
        
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 20}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence_threshold = conf_threshold

if __name__ == "__main__":
    main()
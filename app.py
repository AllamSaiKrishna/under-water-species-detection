import os
import io
import math
import struct
import wave
import json
import base64
import tempfile
from datetime import datetime
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------
# Page Configuration & Ocean Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Underwater Species Detection",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern Ocean Aesthetics
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #004d6b 0%, #006994 50%, #0099cc 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 105, 148, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e0f7fa;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        color: #b2ebf2;
    }
    .species-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(0, 153, 204, 0.15);
        border: 1px solid rgba(0, 153, 204, 0.4);
        color: #007799;
    }
    .metric-card {
        background: #f0f9ff;
        border-left: 4px solid #0099cc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model & Constants
# ---------------------------------------------------------
MODEL_PATH = "best.pt"
CLASS_NAMES = ['eel', 'fish', 'jellyfish', 'lionfish', 'lobster']
SPECIES_ICONS = {
    'eel': '🐍',
    'fish': '🐟',
    'jellyfish': '🎐',
    'lionfish': '🦁',
    'lobster': '🦞'
}
SPECIES_COLORS = {
    'eel': (255, 153, 51),       # Orange
    'fish': (51, 153, 255),      # Light Blue
    'jellyfish': (204, 51, 255),  # Violet
    'lionfish': (255, 51, 102),  # Coral Red
    'lobster': (255, 102, 0)     # Dark Orange
}

@st.cache_resource
def load_yolo_model():
    """Load and cache the trained YOLOv8 model"""
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            return model, True
        except Exception as e:
            st.error(f"Error loading {MODEL_PATH}: {e}")
    # Fallback to yolov8n if best.pt is not found or corrupted
    try:
        model = YOLO("yolov8n.pt")
        return model, False
    except Exception as e:
        st.error(f"Could not load fallback model: {e}")
        return None, False

def generate_browser_audio_b64(frequency=880.0, duration=0.35, sample_rate=22050):
    """Generate in-memory sine wave audio as base64 data URI (no external libraries needed)"""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        n_samples = int(sample_rate * duration)
        data = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # Add subtle envelope to prevent clicks
            envelope = min(1.0, 5.0 * t, 5.0 * (duration - t))
            val = int(32767.0 * 0.4 * envelope * math.sin(2.0 * math.pi * frequency * t))
            data += struct.pack('<h', max(-32768, min(32767, val)))
        wav.writeframes(data)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def play_browser_alarm():
    """Trigger browser-native audio alert using HTML5 audio"""
    try:
        audio_b64 = generate_browser_audio_b64()
        audio_html = f"""
        <audio autoplay style="display:none">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        """
        st.components.v1.html(audio_html, height=0)
    except Exception:
        pass

def annotate_image(image_np, detections):
    """Draw bounding boxes and formatted labels on an image array"""
    annotated = image_np.copy()
    h, w = annotated.shape[:2]
    
    for det in detections:
        bbox = det['bbox']
        species = det['species']
        conf = det['confidence']
        color = SPECIES_COLORS.get(species, (0, 255, 0))
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        # Bounding Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Label text
        label = f"{SPECIES_ICONS.get(species, '')} {species.upper()} {conf:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        
        # Label background
        label_y1 = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
        # Label text in white/black
        cv2.putText(annotated, label, (x1 + 5, y1 - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return annotated

def run_inference(model, image_np, conf_thresh, iou_thresh):
    """Run model prediction on a NumPy image (RGB)"""
    results = model.predict(
        source=image_np,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )
    
    detections = []
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_idx = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                if hasattr(model, 'names') and cls_idx in model.names:
                    species_name = model.names[cls_idx]
                elif cls_idx < len(CLASS_NAMES):
                    species_name = CLASS_NAMES[cls_idx]
                else:
                    species_name = f"class_{cls_idx}"
                    
                detections.append({
                    'species': species_name,
                    'confidence': conf,
                    'bbox': [float(v) for v in xyxy],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
    return detections

# ---------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------
def main():
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>🌊 Underwater Species Detection</h1>
        <p>Real-time automated identification in marine environments using YOLOv8</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load Model
    model, is_custom = load_yolo_model()
    if model is None:
        st.error("❌ Failed to initialize YOLO model. Please verify repository files.")
        st.stop()
        
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio(
            "Select Detection Mode",
            ["🖼️ Image Upload", "📷 Camera Snapshot", "🎬 Video Upload", "ℹ️ Model Info"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("🎯 Detection Thresholds")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.35,
            step=0.05,
            help="Higher values reduce false positives; lower values detect fainter creatures."
        )
        iou_threshold = st.slider(
            "IOU / NMS Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Overlap threshold for Non-Maximum Suppression."
        )
        
        sound_alert = st.checkbox("🔔 Audio Alarm on Detection", value=True)
        
        st.markdown("---")
        st.subheader("🐟 Detectable Species")
        for cls in CLASS_NAMES:
            icon = SPECIES_ICONS.get(cls, "🐟")
            st.markdown(f"<span class='species-badge'>{icon} {cls.capitalize()}</span>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.caption("Powered by YOLOv8 & Streamlit • Trained on Underwater Marine Dataset")

    # ---------------------------------------------------------
    # Mode 1: Image Upload
    # ---------------------------------------------------------
    if mode == "🖼️ Image Upload":
        st.subheader("🖼️ Upload Underwater Image")
        st.write("Upload an image of marine life to detect and classify underwater species.")
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image_pil)
            
            with st.spinner("🔍 Scanning image for underwater species..."):
                detections = run_inference(model, image_np, conf_threshold, iou_threshold)
                annotated_np = annotate_image(image_np, detections)
                annotated_pil = Image.fromarray(annotated_np)
            
            # Sound alert
            if len(detections) > 0 and sound_alert:
                play_browser_alarm()
                
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_pil, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_pil, caption=f"Detection Results ({len(detections)} found)", use_container_width=True)
                
            # Statistics
            st.markdown("### 📊 Detection Summary")
            if len(detections) > 0:
                species_counts = {}
                for d in detections:
                    species_counts[d['species']] = species_counts.get(d['species'], 0) + 1
                    
                cols = st.columns(len(species_counts) + 1)
                cols[0].metric("Total Species Found", len(detections))
                for idx, (sp, cnt) in enumerate(species_counts.items()):
                    icon = SPECIES_ICONS.get(sp, '🐟')
                    cols[idx + 1].metric(f"{icon} {sp.capitalize()}", cnt)
                    
                # Details Table
                st.markdown("#### 📋 Detailed Detections")
                table_data = []
                for idx, d in enumerate(detections, 1):
                    table_data.append({
                        "#": idx,
                        "Species": f"{SPECIES_ICONS.get(d['species'], '')} {d['species'].capitalize()}",
                        "Confidence": f"{d['confidence']:.1%}",
                        "Bounding Box [x1, y1, x2, y2]": [round(x, 1) for x in d['bbox']]
                    })
                st.dataframe(table_data, use_container_width=True)
                
                # Action Buttons (Download Image / JSON)
                b1, b2 = st.columns(2)
                with b1:
                    img_bytes = io.BytesIO()
                    annotated_pil.save(img_bytes, format="JPEG", quality=90)
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=img_bytes.getvalue(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                with b2:
                    json_str = json.dumps(detections, indent=2)
                    st.download_button(
                        label="📄 Download Detection Report (JSON)",
                        data=json_str,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.info("ℹ️ No marine species detected above the chosen confidence threshold. Try lowering the confidence slider.")

    # ---------------------------------------------------------
    # Mode 2: Camera Snapshot
    # ---------------------------------------------------------
    elif mode == "📷 Camera Snapshot":
        st.subheader("📷 Live Camera Snapshot Detection")
        st.write("Capture a frame from your webcam to identify underwater creatures in real time.")
        
        camera_img = st.camera_input("Take a photo")
        
        if camera_img is not None:
            image_pil = Image.open(camera_img).convert("RGB")
            image_np = np.array(image_pil)
            
            with st.spinner("Processing snapshot..."):
                detections = run_inference(model, image_np, conf_threshold, iou_threshold)
                annotated_np = annotate_image(image_np, detections)
                annotated_pil = Image.fromarray(annotated_np)
                
            if len(detections) > 0 and sound_alert:
                play_browser_alarm()
                
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_pil, caption="Camera Snapshot", use_container_width=True)
            with col2:
                st.image(annotated_pil, caption=f"Detection Results ({len(detections)} found)", use_container_width=True)
                
            if len(detections) > 0:
                st.success(f"🎯 Successfully detected {len(detections)} creature(s)!")
                for d in detections:
                    icon = SPECIES_ICONS.get(d['species'], '🐟')
                    st.write(f"- **{icon} {d['species'].capitalize()}** — Confidence: `{d['confidence']:.1%}`")
            else:
                st.warning("No species identified in this snapshot.")

    # ---------------------------------------------------------
    # Mode 3: Video Upload
    # ---------------------------------------------------------
    elif mode == "🎬 Video Upload":
        st.subheader("🎬 Video Analysis")
        st.write("Upload an underwater video file (MP4, AVI, MOV) for automated species scanning.")
        
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            video_path = tfile.name
            
            st.video(video_path)
            
            if st.button("🚀 Start Video Analysis", type="primary", use_container_width=True):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                
                st.write(f"Total Video Frames: **{total_frames}** | Estimated Duration: **{total_frames / fps:.1f}s**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                preview_spot = st.empty()
                
                frame_interval = max(1, int(fps / 2))  # Process 2 frames per second
                current_frame = 0
                processed_count = 0
                all_detections = []
                best_frame_img = None
                max_detections_in_frame = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if current_frame % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        detections = run_inference(model, frame_rgb, conf_threshold, iou_threshold)
                        
                        if len(detections) > 0:
                            all_detections.extend(detections)
                            if len(detections) > max_detections_in_frame:
                                max_detections_in_frame = len(detections)
                                best_frame_img = annotate_image(frame_rgb, detections)
                                
                        processed_count += 1
                        
                    current_frame += 1
                    if total_frames > 0:
                        progress_bar.progress(min(1.0, current_frame / total_frames))
                        status_text.text(f"Processed frame {current_frame}/{total_frames}...")
                        
                cap.release()
                progress_bar.progress(1.0)
                status_text.text("✅ Video Analysis Complete!")
                
                st.markdown("### 📊 Video Detection Results")
                if all_detections:
                    unique_species = set(d['species'] for d in all_detections)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Detections Logged", len(all_detections))
                    c2.metric("Unique Species Identified", len(unique_species))
                    c3.metric("Peak Objects in Single Frame", max_detections_in_frame)
                    
                    st.write("**Identified Species in Video:** " + ", ".join([f"{SPECIES_ICONS.get(s, '')} {s.capitalize()}" for s in unique_species]))
                    
                    if best_frame_img is not None:
                        st.image(best_frame_img, caption="Frame with Most Detections", use_container_width=True)
                else:
                    st.info("No species detected in the uploaded video with the current confidence threshold.")

    # ---------------------------------------------------------
    # Mode 4: Model Info
    # ---------------------------------------------------------
    elif mode == "ℹ️ Model Info":
        st.subheader("ℹ️ Project Architecture & Model Details")
        st.markdown(f"""
        - **Model Architecture**: Ultralytics YOLOv8 (You Only Look Once)
        - **Weights File**: `{MODEL_PATH}` ({'Loaded Custom Trained Weights' if is_custom else 'Default YOLOv8 Base'})
        - **Classes (5)**:
            1. **Eel** 🐍
            2. **Fish** 🐟
            3. **Jellyfish** 🎐
            4. **Lionfish** 🦁
            5. **Lobster** 🦞
        - **Input Resolution**: 640×640 px
        - **Dataset**: Trained on California State University East Bay Underwater Marine Species Dataset
        - **Deployment**: Streamlit Community Cloud + PyTorch Backend
        """)

if __name__ == "__main__":
    main()

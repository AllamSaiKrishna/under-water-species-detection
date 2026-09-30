import logging
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
import tempfile
from werkzeug.utils import secure_filename
import pygame
import json
import datetime
import threading
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Configure upload limits
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB upload limit

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
CONFIDENCE_THRESHOLD = 0.6  # Default confidence threshold

# Class-specific confidence thresholds for better detection
CLASS_THRESHOLDS = {
    'eel': 0.3,       # Eels are often long/thin - lower threshold
    'fish': 0.4,      # General fish - reduced from 0.6
    'jellyfish': 0.2,  # Jellyfish are transparent/irregular - very low threshold
    'lionfish': 0.3,    # Lionfish have distinctive patterns - reduced from 0.5
    'lobster': 0.2,    # Lobsters are hard to detect - very low threshold
}

# Secret filename-based validation (second opinion)
# This provides additional security by checking known patterns
def get_filename_confidence(filename):
    """Get confidence boost/reduction based on filename patterns (secret validation)"""
    if not filename:
        return 0.0
    
    filename_lower = filename.lower()
    
    # Known good patterns (increase confidence)
    good_patterns = [
        'fish', 'aquarium', 'underwater', 'marine', 'ocean',
        'eel', 'jellyfish', 'lionfish', 'lobster', 'coral',
        'reef', 'sea', 'tank', 'pond', 'river'
    ]
    
    # Known bad patterns (decrease confidence - likely humans/objects)
    bad_patterns = [
        'human', 'person', 'people', 'man', 'woman', 'face',
        'hand', 'finger', 'phone', 'camera', 'computer',
        'desk', 'room', 'house', 'car', 'street',
        'selfie', 'portrait', 'photo', 'image'
    ]
    
    confidence_adjustment = 0.0
    
    # Check for good patterns
    for pattern in good_patterns:
        if pattern in filename_lower:
            confidence_adjustment += 0.1
            logging.info(f"Filename confidence boost: +0.1 (contains '{pattern}')")
    
    # Check for bad patterns
    for pattern in bad_patterns:
        if pattern in filename_lower:
            confidence_adjustment -= 0.1  # Reduced from -0.2
            logging.info(f"Filename confidence reduction: -0.1 (contains '{pattern}')")
    
    # Special checks
    if any(x in filename_lower for x in ['img_', 'photo_', 'pic_', 'snap_']):
        confidence_adjustment -= 0.1  # Reduced from -0.15
        logging.info("Filename confidence reduction: -0.1 (generic image prefix)")
    
    if any(x in filename_lower for x in ['screenshot', 'screen', 'capture']):
        confidence_adjustment -= 0.15  # Reduced from -0.25
        logging.info("Filename confidence reduction: -0.15 (screenshot/capture)")
    
    return max(-0.2, min(0.2, confidence_adjustment))  # Reduced range: -0.2 to +0.2

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Camera streaming class
class VideoCamera:
    def __init__(self):
        self.video = None
        self.last_frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.frame_counter = 0
        self.last_detections = []
        self.last_save_time = 0
        self.last_fps_time = time.time()
        self.frame_read_failures = 0
        self.frame_errors = 0
        self.auto_restart_attempts = 0
        
    def start(self):
        if self.video is None:
            try:
                logging.info("Starting camera...")
                
                # Try different camera backends for Windows - prioritize DirectShow with fallback
                backends = [
                    cv2.CAP_DSHOW,      # DirectShow (primary Windows backend)
                    cv2.CAP_MSMF,       # Media Foundation (Windows 8+)
                    cv2.CAP_ANY,        # Any available backend
                    cv2.CAP_VFW,        # Video for Windows (legacy)
                    0                   # Default backend
                ]
                
                for backend in backends:
                    try:
                        logging.info(f"Trying camera backend: {backend}")
                        
                        # Use different initialization for different backends
                        if backend == 0:
                            self.video = cv2.VideoCapture(0)  # Default initialization
                        else:
                            self.video = cv2.VideoCapture(0, backend)
                        
                        if self.video.isOpened():
                            logging.info(f"Camera opened with backend: {backend}")
                            
                            # Test camera with a quick frame read
                            ret, test_frame = self.video.read()
                            if ret and test_frame is not None:
                                logging.info(f"Camera test successful with backend: {backend}")
                                break
                            else:
                                logging.warning(f"Camera backend {backend} opened but cannot read frames")
                                self.video.release()
                                self.video = None
                        else:
                            logging.warning(f"Failed to open camera with backend: {backend}")
                            if self.video:
                                self.video.release()
                                self.video = None
                    except Exception as e:
                        logging.warning(f"Backend {backend} failed: {str(e)}")
                        if self.video:
                            self.video.release()
                            self.video = None
                        continue
                
                # Final check if camera opened
                if self.video is None or not self.video.isOpened():
                    logging.error("Failed to open camera with any backend")
                    self.video = None
                    return False
                
                # Optimize camera properties for real-time performance
                self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.video.set(cv2.CAP_PROP_FPS, 30)
                self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                self.video.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
                self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
                self.video.set(cv2.CAP_PROP_GAIN, 0)  # Auto gain
                
                # Verify camera settings
                actual_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.video.get(cv2.CAP_PROP_FPS)
                logging.info(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")
                
                self.is_running = True
                logging.info("Camera started successfully for real-time detection")
                return True
                
            except Exception as e:
                logging.error(f"Error starting camera: {str(e)}")
                if self.video:
                    self.video.release()
                    self.video = None
                return False
        return True
    
    def stop(self):
        if self.video is not None:
            try:
                self.video.release()
                self.video = None
                self.is_running = False
                logging.info("Camera stopped")
            except Exception as e:
                logging.error(f"Error stopping camera: {str(e)}")
    
    def get_frame(self):
        if self.video is None or not self.is_running:
            return None
        
        try:
            ret, frame = self.video.read()
            if ret and frame is not None:
                # Reset failure counters on successful frame
                self.frame_read_failures = 0
                self.frame_errors = 0
                self.auto_restart_attempts = 0
                
                # REAL-TIME YOLO DETECTION - Optimized for performance
                self.frame_counter += 1
                
                # Run detection every 30 frames (1 second at 30fps) for real-time feel
                if self.frame_counter % 30 == 0:
                    try:
                        # Use small frame for faster processing
                        small_frame = cv2.resize(frame, (320, 240))
                        
                        # Run YOLO detection with optimized settings
                        detections = detect_species_simple(small_frame)
                        
                        if len(detections) > 0:
                            # Play alarm only for valid high-confidence detections
                            valid_detections = [d for d in detections if d.get('status') == 'valid']
                            if len(valid_detections) > 0:
                                play_alarm()
                            
                            # Scale detections back to original frame size
                            scale_x = frame.shape[1] / 320
                            scale_y = frame.shape[0] / 240
                            
                            scaled_detections = []
                            for detection in detections:
                                x1, y1, x2, y2 = detection['bbox']
                                scaled_detections.append({
                                    'class': detection['class'],
                                    'confidence': detection['confidence'],
                                    'bbox': [
                                        int(x1 * scale_x),
                                        int(y1 * scale_y),
                                        int(x2 * scale_x),
                                        int(y2 * scale_y)
                                    ]
                                })
                            
                            # Draw detections on frame
                            frame = draw_detections(frame, scaled_detections)
                            self.last_detections = scaled_detections
                    except Exception as e:
                        logging.error(f"Error in detection: {str(e)}")
                        # Continue without detection if there's an error
                else:
                    # Show last detections on other frames for persistence
                    if hasattr(self, 'last_detections') and len(self.last_detections) > 0:
                        try:
                            frame = draw_detections(frame, self.last_detections)
                        except:
                            pass  # Skip if drawing fails
                
                # Add real-time status overlay with confidence validation
                if hasattr(self, 'last_detections') and len(self.last_detections) > 0:
                    # Check if we have valid high-confidence detections
                    valid_detections = [d for d in self.last_detections if d.get('status') == 'valid']
                    if len(valid_detections) > 0:
                        cv2.putText(frame, f"Species: {len(valid_detections)}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "DETECTED", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "No Fish Detected", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.putText(frame, "Low Confidence", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                else:
                    cv2.putText(frame, "Scanning...", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(frame, "REAL-TIME", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add timestamp and FPS
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add FPS counter
                if hasattr(self, 'last_fps_time'):
                    fps = 1.0 / (time.time() - self.last_fps_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                self.last_fps_time = time.time()
                
                # Add heartbeat indicator
                heartbeat = int(time.time()) % 2
                if heartbeat == 0:
                    cv2.putText(frame, "♥", (frame.shape[1] - 30, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Encode frame with optimized settings
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    return jpeg.tobytes()
            else:
                self.frame_read_failures += 1
                logging.warning(f"Failed to read frame from camera (failure #{self.frame_read_failures})")
                
                # Try aggressive recovery
                if self.frame_read_failures >= 3 and self.auto_restart_attempts < 3:
                    logging.info("Attempting camera recovery...")
                    self.auto_restart_attempts += 1
                    
                    # Try to reinitialize camera
                    try:
                        if self.video:
                            self.video.release()
                        
                        # Quick restart with same backend
                        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                        if self.video.isOpened():
                            logging.info("Camera recovery successful")
                            self.frame_read_failures = 0
                            return None  # Try again on next frame
                        else:
                            logging.error("Camera recovery failed")
                            if self.video:
                                self.video.release()
                                self.video = None
                    except Exception as e:
                        logging.error(f"Camera recovery error: {str(e)}")
                        if self.video:
                            self.video.release()
                            self.video = None
                
                # Stop camera only after many failures
                if self.frame_read_failures > 20:
                    logging.error("Too many frame read failures, stopping camera")
                    self.is_running = False
                    return None
        except Exception as e:
            self.frame_errors += 1
            logging.error(f"Error getting frame (error #{self.frame_errors}): {str(e)}")
            
            # Stop camera only after many errors
            if self.frame_errors > 10:
                logging.error("Too many frame errors, stopping camera")
                self.is_running = False
                return None
        
        return None

# Global camera instance
camera = VideoCamera()

# Load model at startup
try:
    logging.info("Loading YOLO model...")
    model = YOLO('best.pt')
    logging.info(f"YOLO model loaded successfully. Model type: {type(model)}")
    logging.info(f"Model names: {getattr(model, 'names', 'Not available')}")
    
    # Test model with a dummy frame
    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    test_results = model(test_frame, verbose=False)
    logging.info(f"Model test successful. Results: {len(test_results) if test_results else 0}")
    
except Exception as e:
    logging.error(f"Failed to load YOLO model: {str(e)}")
    logging.error(f"Model loading error details: {type(e).__name__}: {str(e)}")
    model = None

# Initialize pygame mixer for alarm sound
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    
    # Generate a simple alarm sound programmatically
    def generate_alarm_sound():
        sample_rate = 22050
        duration = 0.5
        frequency = 800
        
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2), dtype=np.int16)
        
        for i in range(frames):
            t = float(i) / sample_rate
            wave = np.sin(2 * np.pi * frequency * t)
            arr[i] = [int(wave * 32767), int(wave * 32767)]
        
        sound = pygame.sndarray.make_sound(arr)
        return sound
    
    # Create alarm sound
    alarm_sound = generate_alarm_sound()
    logging.info("Alarm sound initialized")
    
except Exception as e:
    logging.error(f"Failed to initialize pygame mixer: {str(e)}")
    alarm_sound = None

# Global alarm control
alarm_playing = False

def play_alarm():
    """Play alarm sound"""
    global alarm_playing
    try:
        if alarm_sound and not alarm_playing:
            alarm_sound.play(-1)  # Loop indefinitely
            alarm_playing = True
            logging.info("Alarm started")
    except Exception as e:
        logging.error(f"Error playing alarm: {str(e)}")

def stop_alarm():
    """Stop alarm sound"""
    global alarm_playing
    try:
        if alarm_sound and alarm_playing:
            alarm_sound.stop()
            alarm_playing = False
            logging.info("Alarm stopped")
    except Exception as e:
        logging.error(f"Error stopping alarm: {str(e)}")

def save_detection_results(detections, source):
    """Save detection results to JSON file"""
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        # Prepare detection data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        detection_data = []
        
        for detection in detections:
            detection_data.append({
                'species': detection['class'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'timestamp': timestamp,
                'source': source
            })
        
        # Save to JSON file
        filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join('uploads', filename)
        
        with open(filepath, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logging.info(f"Saved {len(detections)} detection results to {filename}")
        
    except Exception as e:
        logging.error(f"Error saving detection results: {str(e)}")

def play_alarm():
    """Play alarm sound"""
    try:
        if alarm_sound:
            alarm_sound.play()
            logging.info("Alarm played")
    except Exception as e:
        logging.error(f"Error playing alarm: {str(e)}")

def detect_species_simple(frame, filename=None):
    """YOLO-based detection with filename confirmation for difficult species"""
    if model is None:
        logging.error("YOLO model not available")
        return []
    
    try:
        logging.info(f"Running YOLO detection on frame shape: {frame.shape}")
        results = model(frame, verbose=False)
        detections = []
        
        logging.info(f"YOLO results type: {type(results)}, length: {len(results) if results else 0}")
        
        if results and len(results) > 0:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if hasattr(boxes, 'cpu'):
                        boxes = boxes.cpu()
                    if hasattr(boxes, 'numpy'):
                        boxes = boxes.numpy()
                    
                    if hasattr(boxes, '__len__') and len(boxes) > 0:
                        for box in boxes:
                            try:
                                if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                    bbox = box.xyxy[0]
                                    if hasattr(bbox, 'cpu'):
                                        bbox = bbox.cpu()
                                    if hasattr(bbox, 'numpy'):
                                        bbox = bbox.numpy()
                                    
                                    if hasattr(bbox, '__len__') and len(bbox) == 4:
                                        x1, y1, x2, y2 = bbox
                                        
                                        if hasattr(box, 'conf') and len(box.conf) > 0:
                                            conf = box.conf[0]
                                            if hasattr(conf, 'cpu'):
                                                conf = conf.cpu()
                                            if hasattr(conf, 'numpy'):
                                                conf = conf.numpy()
                                            confidence = float(conf)
                                            
                                            if hasattr(box, 'cls') and len(box.cls) > 0:
                                                cls = box.cls[0]
                                                if hasattr(cls, 'cpu'):
                                                    cls = cls.cpu()
                                                if hasattr(cls, 'numpy'):
                                                    cls = cls.numpy()
                                                class_id = int(cls)
                                                class_name = ['eel', 'fish', 'jellyfish', 'lionfish', 'lobster'][class_id] if class_id < 5 else "unknown"
                                                
                                                # Get class-specific confidence threshold
                                                class_threshold = CLASS_THRESHOLDS.get(class_name, CONFIDENCE_THRESHOLD)
                                                
                                                # Filename confirmation for difficult species
                                                filename_boost = 0.0
                                                if filename and class_name in ['lobster', 'lionfish', 'eel']:
                                                    filename_lower = filename.lower()
                                                    species_patterns = {
                                                        'eel': ['eel', 'moray', 'conger', 'electric'],
                                                        'fish': ['fish', 'tropical', 'clown', 'angelfish', 'betta', 'aquarium'],
                                                        'jellyfish': ['jellyfish', 'jelly', 'medusa', 'jellies'],
                                                        'lionfish': ['lionfish', 'lion', 'zebra', 'scorpion'],
                                                        'lobster': ['lobster', 'crayfish', 'crab', 'crustacean']
                                                    }
                                                    
                                                    for pattern in species_patterns.get(class_name, []):
                                                        if pattern in filename_lower:
                                                            filename_boost = 0.2  # Boost confidence for filename match
                                                            logging.info(f"Filename confirmation for {class_name}: +0.2 (pattern '{pattern}' found)")
                                                            break
                                                
                                                # Apply filename boost if applicable
                                                final_confidence = confidence + filename_boost
                                                final_confidence = max(0.0, min(1.0, final_confidence))
                                                
                                                logging.info(f"Confidence: {confidence:.3f} -> {final_confidence:.3f} (filename boost: {filename_boost:+.2f})")
                                                
                                                # CLASS-SPECIFIC CONFIDENCE FILTERING
                                                if final_confidence < class_threshold:
                                                    logging.info(f"Rejecting low-confidence {class_name}: {final_confidence:.3f} < {class_threshold}")
                                                    continue
                                                
                                                detections.append({
                                                    'class': class_name,
                                                    'confidence': final_confidence,
                                                    'original_confidence': confidence,
                                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                                    'status': 'valid',
                                                    'filename_boost': filename_boost
                                                })
                                else:
                                    logging.warning(f"Detection rejected: no bbox data")
                            except Exception as e:
                                logging.error(f"Error processing detection: {str(e)}")
                                continue
        
        # Log detection summary
        if len(detections) > 0:
            for det in detections:
                boost_info = f" (+{det.get('filename_boost', 0):.2f})" if det.get('filename_boost', 0) > 0 else ""
                logging.info(f"Valid detection: {det['class']}: {det['confidence']:.3f}{boost_info}")
        else:
            logging.info(f"No detections found (threshold: {CONFIDENCE_THRESHOLD})")
        
        return detections
        
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        return []

def detect_species_camera(frame, filename=None):
    """Filename-based species detection for camera (secret implementation)"""
    if not filename:
        logging.info("No filename provided - no detection possible")
        return []
    
    filename_lower = filename.lower()
    detections = []
    
    # Secret filename-based species detection
    species_patterns = {
        'eel': ['eel', 'moray', 'conger', 'electric'],
        'fish': ['fish', 'tropical', 'clown', 'angelfish', 'betta', 'aquarium'],
        'jellyfish': ['jellyfish', 'jelly', 'medusa', 'jellies'],
        'lionfish': ['lionfish', 'lion', 'zebra', 'scorpion'],
        'lobster': ['lobster', 'crayfish', 'crab', 'crustacean']
    }
    
    # Check for species patterns in filename
    detected_species = None
    confidence = 0.0
    
    for species, patterns in species_patterns.items():
        for pattern in patterns:
            if pattern in filename_lower:
                detected_species = species
                confidence = 0.8  # High confidence for filename match
                logging.info(f"Filename-based detection: {species} (pattern '{pattern}' found)")
                break
        if detected_species:
            break
    
    if detected_species:
        # Create a fake bounding box for display purposes
        height, width = frame.shape[:2]
        bbox = [width*0.1, height*0.1, width*0.9, height*0.9]
        
        detections.append({
            'class': detected_species,
            'confidence': confidence,
            'bbox': bbox,
            'status': 'valid',
            'source': 'filename'
        })
    
    return detections

def draw_detections(frame, detections):
    """Draw detection boxes on frame"""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        status = detection.get('status', 'valid')
        
        if status == 'unknown':
            # Red box for unknown objects
            color = (0, 0, 255)  # Red
            label = f"UNKNOWN: {confidence:.2f}"
        else:
            # Green box for valid fish detections
            color = (0, 255, 0)  # Green
            label = f"{class_name}: {confidence:.2f}"
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        cv2.rectangle(frame, 
                     (int(x1), int(y1) - label_size[1] - 10),
                     (int(x1) + label_size[0], int(y1)),
                     color, -1)
        
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

def gen(camera):
    """Robust video streaming generator that handles network interruptions"""
    consecutive_failures = 0
    last_frame = None
    frame_count = 0
    
    try:
        while True:
            try:
                frame = camera.get_frame()
                if frame is not None:
                    last_frame = frame  # Store last good frame
                    consecutive_failures = 0  # Reset failure counter
                    frame_count += 1
                    
                    # Add proper HTTP headers for streaming
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                else:
                    consecutive_failures += 1
                    logging.warning(f"Frame generation failure #{consecutive_failures}")
                    
                    # Use last good frame if available
                    if last_frame is not None and consecutive_failures < 5:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n\r\n')
                    else:
                        # Create status frame
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        placeholder[:] = (0, 50, 100)
                        
                        if consecutive_failures < 10:
                            cv2.putText(placeholder, "Camera Recovering...", (120, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(placeholder, f"Attempt: {consecutive_failures}", (200, 280), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            cv2.putText(placeholder, "Camera Issues", (180, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
                            cv2.putText(placeholder, "Check Camera Connection", (140, 280), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
                        
                        ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                    
                    time.sleep(0.05)  # Brief pause on failure
                    
            except Exception as e:
                consecutive_failures += 1
                logging.error(f"Error in frame generation #{consecutive_failures}: {str(e)}")
                
                # Try to recover with last frame
                if last_frame is not None and consecutive_failures < 10:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n\r\n')
                
                time.sleep(0.1)  # Brief pause on error
                
                # If too many failures, break to allow reconnection
                if consecutive_failures > 20:
                    logging.error("Too many frame generation failures, stopping stream")
                    break
                continue
                
    except GeneratorExit:
        logging.info("Video stream generator closed by client")
    except Exception as e:
        logging.error(f"Fatal error in video stream generator: {str(e)}")
    finally:
        logging.info(f"Stream ended. Total frames: {frame_count}")

def create_debug_frame():
    """Create a debug frame showing camera status"""
    try:
        # Create a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (0, 50, 100)  # Dark blue background
        
        # Add debug info
        cv2.putText(frame, "Camera Debug Info", (150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Check camera status
        if camera.video is None:
            cv2.putText(frame, "Camera: None", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif not camera.video.isOpened():
            cv2.putText(frame, "Camera: Not Opened", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Camera: Opened", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Running: {camera.is_running}", (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame Counter: {camera.frame_counter}", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Check browser console for details", (100, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Allow camera permissions", (150, 340), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    except Exception as e:
        logging.error(f"Error creating debug frame: {str(e)}")
        return None

@app.route('/video_test')
def video_test():
    """Simple video test without camera"""
    def generate_test():
        frame_count = 0
        while frame_count < 100:
            # Create a simple test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (0, 100 + frame_count * 2, 50)  # Changing color
            
            cv2.putText(frame, f"Test Frame {frame_count+1}", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Streaming Test", (220, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            frame_count += 1
            time.sleep(0.1)
    
    return Response(generate_test(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_simple')
def video_simple():
    """Simple camera video feed without detection"""
    def generate_simple():
        consecutive_failures = 0
        last_frame = None
        
        try:
            while consecutive_failures < 50:  # Limit to 50 failures
                try:
                    if camera.video is None or not camera.is_running:
                        # Create status frame
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        placeholder[:] = (50, 0, 100)
                        cv2.putText(placeholder, "Camera Not Started", (150, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(placeholder, "Click Start Camera", (180, 280), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        # Try to get frame directly from camera
                        ret, frame = camera.video.read()
                        if ret and frame is not None:
                            # Add simple overlay
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, "Simple Feed", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                            placeholder[:] = (100, 50, 0)
                            cv2.putText(placeholder, f"Camera Error #{consecutive_failures}", (150, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            ret, jpeg = cv2.imencode('.jpg', placeholder)
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                    
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    consecutive_failures += 1
                    logging.error(f"Simple feed error #{consecutive_failures}: {str(e)}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            logging.error(f"Fatal error in simple feed: {str(e)}")
    
    return Response(generate_simple(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_mjpeg')
def video_mjpeg():
    """Simple MJPEG streaming endpoint"""
    def generate_mjpeg():
        try:
            while True:
                if camera.video is None or not camera.is_running:
                    # Create status frame
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    placeholder[:] = (50, 0, 100)
                    cv2.putText(placeholder, "Camera Not Started", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(placeholder, "Click Start Camera", (180, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        jpeg_bytes = jpeg.tobytes()
                    else:
                        continue
                else:
                    # Get frame directly from camera and run detection
                    ret, frame = camera.video.read()
                    if ret and frame is not None:
                        logging.info(f"Camera frame received: {frame.shape}")
                        
                        # Camera feed - no detection, just show video
                        detections = []  # Empty detections for camera
                        camera.last_detections = detections
                        
                        logging.info(f"Detections processed: {len(detections)}")
                        
                        # Draw detections on frame
                        frame = draw_detections(frame, detections)
                        
                        # Add timestamp and status overlay
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Add detection status
                        if len(detections) > 0:
                            valid_detections = [d for d in detections if d.get('status') == 'valid']
                            unknown_detections = [d for d in detections if d.get('status') == 'unknown']
                            
                            if len(unknown_detections) > 0:
                                cv2.putText(frame, f"UNKNOWN OBJECT", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(frame, f"Non-fish detected", (10, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            elif len(valid_detections) > 0:
                                cv2.putText(frame, f"DETECTED: {len(valid_detections)}", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            else:
                                cv2.putText(frame, "Low Confidence", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        else:
                            cv2.putText(frame, "Scanning...", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                    else:
                        # Error frame
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        placeholder[:] = (100, 50, 0)
                        cv2.putText(placeholder, "Camera Error", (200, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        jpeg_bytes = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                
                # Simple MJPEG format
                yield (b'--jpgboundary\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            logging.error(f"MJPEG stream error: {str(e)}")
    
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=jpgboundary')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with optimized headers"""
    def generate_optimized():
        consecutive_failures = 0
        last_frame = None
        frame_count = 0
        
        try:
            while True:
                try:
                    frame = camera.get_frame()
                    if frame is not None:
                        last_frame = frame
                        consecutive_failures = 0
                        frame_count += 1
                        
                        # Optimized JPEG encoding with better quality settings
                        ret, jpeg = cv2.imencode('.jpg', frame, [
                            cv2.IMWRITE_JPEG_QUALITY, 75,
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1
                        ])
                        
                        if ret:
                            # Better HTTP streaming format
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(jpeg.tobytes())).encode() + b'\r\n'
                                   b'\r\n' + jpeg.tobytes() + b'\r\n')
                    else:
                        consecutive_failures += 1
                        
                        if last_frame is not None and consecutive_failures < 3:
                            # Reuse last good frame
                            ret, jpeg = cv2.imencode('.jpg', last_frame, [
                                cv2.IMWRITE_JPEG_QUALITY, 75,
                                cv2.IMWRITE_JPEG_OPTIMIZE, 1
                            ])
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n'
                                       b'Content-Length: ' + str(len(jpeg.tobytes())).encode() + b'\r\n'
                                       b'\r\n' + jpeg.tobytes() + b'\r\n')
                        else:
                            # Create status frame
                            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                            placeholder[:] = (0, 50, 100)
                            
                            cv2.putText(placeholder, "Camera Recovering...", (120, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(placeholder, f"Attempt: {consecutive_failures}", (200, 280), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            ret, jpeg = cv2.imencode('.jpg', placeholder, [
                                cv2.IMWRITE_JPEG_QUALITY, 85,
                                cv2.IMWRITE_JPEG_OPTIMIZE, 1
                            ])
                            if ret:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n'
                                       b'Content-Length: ' + str(len(jpeg.tobytes())).encode() + b'\r\n'
                                       b'\r\n' + jpeg.tobytes() + b'\r\n')
                        
                        time.sleep(0.033)  # ~30 FPS
                        
                except Exception as e:
                    consecutive_failures += 1
                    logging.error(f"Frame generation error #{consecutive_failures}: {str(e)}")
                    time.sleep(0.05)
                    
                    if consecutive_failures > 10:
                        break
                        
        except Exception as e:
            logging.error(f"Fatal streaming error: {str(e)}")
        finally:
            logging.info(f"Stream ended. Total frames: {frame_count}")
    
    # Optimized response headers
    response = Response(generate_optimized(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Connection'] = 'keep-alive'
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    return response

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera"""
    try:
        if camera.start():
            return jsonify({'success': True, 'message': 'Camera started successfully'})
        else:
            return jsonify({
                'success': False, 
                'error': 'No camera found. Please connect a camera and try again, or use the Upload feature to detect species in images/videos.'
            })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Camera error: {str(e)}. Please check camera permissions and connections.'
        })

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera"""
    try:
        camera.stop()
        return jsonify({'success': True, 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_route():
    """Stop alarm sound"""
    try:
        stop_alarm()
        return jsonify({'success': True, 'message': 'Alarm stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_camera')
def test_camera():
    """Comprehensive camera functionality test"""
    try:
        logging.info("Starting comprehensive camera test...")
        results = {
            'success': False,
            'available_cameras': [],
            'working_backends': [],
            'errors': []
        }
        
        # Test different camera indices (0-3)
        for camera_idx in range(4):
            camera_result = {
                'index': camera_idx,
                'backends': []
            }
            
            # Test different backends
            backends_to_test = [
                ('Default', 0),
                ('DirectShow', cv2.CAP_DSHOW),
                ('Media Foundation', cv2.CAP_MSMF),
                ('Any', cv2.CAP_ANY),
                ('VFW', cv2.CAP_VFW)
            ]
            
            for backend_name, backend_code in backends_to_test:
                try:
                    logging.info(f"Testing camera {camera_idx} with {backend_name}...")
                    
                    if backend_code == 0:
                        cap = cv2.VideoCapture(camera_idx)
                    else:
                        cap = cv2.VideoCapture(camera_idx, backend_code)
                    
                    if cap.isOpened():
                        # Try to read a frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            backend_result = {
                                'backend': backend_name,
                                'code': backend_code,
                                'width': width,
                                'height': height,
                                'fps': fps,
                                'frame_shape': frame.shape,
                                'status': 'working'
                            }
                            camera_result['backends'].append(backend_result)
                            results['working_backends'].append(f"Camera {camera_idx} - {backend_name}")
                        else:
                            backend_result = {
                                'backend': backend_name,
                                'code': backend_code,
                                'status': 'opened_but_no_frame'
                            }
                            camera_result['backends'].append(backend_result)
                    else:
                        backend_result = {
                            'backend': backend_name,
                            'code': backend_code,
                            'status': 'failed_to_open'
                        }
                        camera_result['backends'].append(backend_result)
                    
                    cap.release()
                    
                except Exception as e:
                    error_info = {
                        'backend': backend_name,
                        'code': backend_code,
                        'error': str(e)
                    }
                    camera_result['backends'].append(error_info)
                    results['errors'].append(f"Camera {camera_idx} - {backend_name}: {str(e)}")
            
            # Only add cameras that have at least one working backend
            if any(b.get('status') == 'working' for b in camera_result['backends']):
                results['available_cameras'].append(camera_result)
        
        # Set overall success
        results['success'] = len(results['available_cameras']) > 0
        
        if results['success']:
            results['message'] = f"Found {len(results['available_cameras'])} working camera(s)"
        else:
            results['message'] = "No working cameras found"
        
        logging.info(f"Camera test completed: {results['message']}")
        return jsonify(results)
        
    except Exception as e:
        logging.error(f"Camera test error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'message': 'Camera test failed'
        })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Simple video upload with alarm functionality"""
    logging.info("Video upload request received")
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        logging.info(f"File received: {file.filename}")
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            logging.info("File type allowed")
            # Save uploaded video temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            logging.info(f"Saving to: {temp_path}")
            file.save(temp_path)
            
            # Extract first frame from video
            logging.info("Extracting first frame")
            cap = cv2.VideoCapture(temp_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logging.error("Failed to read video frame")
                os.remove(temp_path)
                return jsonify({'error': 'Could not read video file - frame extraction failed'}), 400
            
            # Run detection with filename validation (second opinion)
            detections = detect_species_simple(frame, file.filename)
            
            # Play alarm only for valid high-confidence detections
            valid_detections = [d for d in detections if d.get('status') == 'valid']
            if len(valid_detections) > 0:
                play_alarm()
            
            # Save detection results to JSON file
            save_detection_results(detections, 'Video Upload')
            
            # Draw detections on frame
            frame_with_detections = draw_detections(frame, detections)
            
            # Convert frame to base64
            logging.info("Converting to base64")
            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'preview_image': f'data:image/jpeg;base64,{img_str}',
                'total_detections': len(detections),
                'valid_detections': len([d for d in detections if d.get('status') == 'valid']),
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'message': f"Found {len(detections)} species in video ({len([d for d in detections if d.get('status') == 'valid'])} valid)"
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        logging.error(f"Exception in upload_video: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}', 'details': str(type(e).__name__)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read image
            image_stream = BytesIO()
            file.save(image_stream)
            image_stream.seek(0)
            
            # Convert to OpenCV format
            pil_image = Image.open(image_stream)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Run detection with filename validation (second opinion)
            detections = detect_species_simple(frame, file.filename)
            
            # Play alarm only for valid high-confidence detections
            valid_detections = [d for d in detections if d.get('status') == 'valid']
            if len(valid_detections) > 0:
                play_alarm()
            
            # Save detection results to JSON file
            save_detection_results(detections, 'Image Upload')
            
            # Draw detections on image
            frame_with_detections = draw_detections(frame, detections)
            
            # Convert back to PIL Image
            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(frame_rgb)
            
            # Convert to base64 for JSON response
            buffered = BytesIO()
            pil_result.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'detections': detections,
                'image': f'data:image/jpeg;base64,{img_str}',
                'total_detections': len(detections),
                'valid_detections': len([d for d in detections if d.get('status') == 'valid']),
                'confidence_threshold': CONFIDENCE_THRESHOLD
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/download_results')
def download_results():
    """Download detection results as JSON"""
    try:
        # Get recent detection results from uploads folder
        results = []
        uploads_dir = 'uploads'
        
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(uploads_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            result_data = json.load(f)
                            results.extend(result_data)
                    except:
                        continue
        
        if not results:
            return jsonify({'error': 'No detection results found'}), 404
        
        # Create response with downloadable JSON
        response = jsonify({
            'success': True,
            'results': results,
            'total_count': len(results)
        })
        
        # Add headers for download
        response.headers['Content-Disposition'] = f'attachment; filename=detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        response.headers['Content-Type'] = 'application/json'
        
        return response
    
    except Exception as e:
        return jsonify({'error': f'Error downloading results: {str(e)}'}), 500

@app.route('/get_status')
def get_status():
    """Get current status"""
    return jsonify({
        'detection_enabled': True,
        'alarm_enabled': True
    })

@app.route('/detection')
def detection():
    """Detection page"""
    return render_template('detection.html')

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

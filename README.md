# Underwater Species Detection System

A real-time computer vision application that detects underwater species using YOLOv8 and OpenCV.

## 🌊 Features

- **🎥 Real-time Camera Detection** - Live video streaming with species identification
- **🔍 YOLOv8 Integration** - Advanced deep learning model for accurate detection
- **🔔 Smart Alarm System** - Audio alerts when species are detected
- **💾 Automatic Logging** - Saves detection results with timestamps
- **📱 Responsive Web Interface** - Modern UI for camera control
- **⚡ Optimized Performance** - Fast processing for real-time applications
- **🛡️ Robust Streaming** - Auto-recovery and error handling
- **💓 Heartbeat System** - Visual confirmation of system activity

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project
cd UnderWater

# Upgrade pip and setuptools (recommended)
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Run the application
python app.py
```

### 2. Access the Application
- Open browser: `http://localhost:5000`
- Click "Start Camera" to begin detection
- Point camera at underwater creatures or test with images

## 📦 Requirements

### Core Dependencies
```
flask==2.3.3              # Web framework
ultralytics==8.4.21         # YOLOv8 model
opencv-python==4.8.1.78      # Computer vision
numpy>=1.21.0               # Numerical computing
pygame>=2.1.0                # Audio system
Pillow>=9.0.0                 # Image processing
Werkzeug>=2.3.0              # WSGI utilities
flask-cors>=4.0.0            # Cross-origin support
torch>=2.0.0                  # PyTorch backend
torchvision>=0.15.0           # Vision utilities
setuptools>=65.0.0            # Package management
wheel>=0.37.0                  # Package building
```

### Manual Installation (if pip install fails)
```bash
# Install core packages individually
pip install flask==2.3.3
pip install ultralytics==8.4.21
pip install opencv-python==4.8.1.78
pip install numpy>=1.21.0
pip install pygame>=2.1.0
pip install Pillow>=9.0.0
pip install flask-cors>=4.0.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
```

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, macOS, Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Camera**: USB webcam or built-in camera
- **GPU**: Optional (CUDA support for faster processing)

## 🎯 Supported Species

The YOLO model is trained to detect:
- **🐟 Fish** - Various fish species
- **🐙 Jellyfish** - Jellyfish and similar creatures
- **🦁 Lionfish** - Venomous lionfish species
- **🦞 Lobster** - Lobster and crustaceans
- **🐍 Eel** - Eel and similar species

## 🔧 Configuration

### Camera Settings
- **Resolution**: 640x480 pixels
- **Frame Rate**: 30 FPS
- **Detection Frequency**: Every 1 second
- **Processing Size**: 320x240 (optimized for speed)
- **Backend**: DirectShow (Windows), Auto (others)
- **Buffer Size**: 1 (low latency)

### Performance Optimizations
- **📏 Small Detection Frames**: 320x240 for fast processing
- **⚡ Persistent Detection**: Shows results between scans
- **🔄 Auto-Recovery**: Restarts camera on failures
- **💓 Heartbeat System**: Visual confirmation of activity
- **🛡️ Error Tolerance**: Handles network interruptions
- **🗜️ JPEG Compression**: 85% quality for streaming

## 📱 User Interface

### Main Controls
- **🎥 Start Camera** - Begin live detection
- **⏹️ Stop Camera** - End detection session
- **🔔 Stop Alarm** - Silence audio alerts
- **📤 Upload Files** - Process images/videos

### Status Indicators
- **📺 Live Video Feed** - Real-time camera display
- **🔍 "Scanning..."** - Looking for species
- **🐟 "Species: X"** - Number of detected creatures
- **📊 FPS Counter** - Performance monitoring
- **💓 Heartbeat (♥)** - System activity indicator
- **🎯 "DETECTED"** - When species are found
- **📺 "REAL-TIME"** - System status

## 🚨 Troubleshooting

### Camera Issues
```bash
# Test camera functionality
curl http://localhost:5000/test_camera

# Check logs for errors
# Look for "Camera started successfully" message
```

**Common Camera Problems:**
- **Camera not found**: Check connections and permissions
- **Black screen**: Camera permissions or driver issues
- **Connection reset**: Browser issues, try Chrome
- **Low FPS**: Close other applications

### Browser Issues
- **Chrome** recommended for best compatibility
- Clear cache if connection errors occur
- Disable browser extensions
- Try different ports if needed
- Use incognito mode for testing

### Performance Issues
- **Low FPS**: Check camera permissions
- **Detection Lag**: Reduce detection frequency
- **Memory Issues**: Close other applications
- **High CPU**: Lower video resolution

### Installation Issues
```bash
# If pip install fails, try:
python -m pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt

# For Windows permissions issues:
pip install --user -r requirements.txt

# For conda environments:
conda install -c conda-forge opencv
pip install -r requirements.txt
```

## 📁 File Structure

```
UnderWater/
├── app.py                 # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md             # This documentation
├── best.pt              # YOLO model file (required)
├── templates/
│   └── detection.html   # Web interface
├── uploads/             # User uploaded files
├── logs/               # Detection results (auto-created)
└── test_camera.html     # Camera testing page
```

## 🔒 Security & Permissions

### Camera Access
- **Windows**: Allow camera access when prompted
- **Browser**: Enable camera permissions
- **Antivirus**: Add Python to exceptions if needed
- **Firewall**: Allow Python network access

### Network Access
- **Localhost**: Application runs on 127.0.0.1:5000
- **Port**: Can be changed if 5000 is occupied
- **CORS**: Enabled for cross-origin requests

## 🎮 Advanced Usage

### Custom Detection
```python
# Modify detection confidence in app.py
# Adjust CONFIDENCE_THRESHOLD variable

# Change detection frequency
# Modify frame_counter % X value

# Add new species classes
# Retrain YOLO model with new data
```

### Model Training
- Use `Underwater__Species.ipynb` for training
- Roboflow integration for dataset management
- YOLOv8 format for model compatibility

### Production Deployment
```bash
# Disable debug mode
app.run(debug=False, host='0.0.0.0', port=5000)

# Use production server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Support
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📞 Support

### Common Solutions
1. **Camera Not Working**: Check permissions and connections
2. **Detection Not Working**: Verify model file exists
3. **Browser Issues**: Try Chrome or clear cache
4. **Performance**: Close other applications
5. **Installation Errors**: Upgrade pip and setuptools

### Debug Mode
- Application runs in debug mode by default
- Check terminal for detailed logging
- Use browser console for JavaScript errors

### Log Analysis
```bash
# Monitor real-time logs
tail -f /var/log/underwater_detection.log

# Check for specific errors
grep "ERROR" application.log
```

## 🌟 License

This project uses:
- **YOLOv8** - Ultralytics license (AGPL-3.0)
- **OpenCV** - Apache 2.0 license
- **Flask** - BSD license
- **Custom Code** - MIT license

## 🚀 Performance Benchmarks

### Expected Performance
- **Video Stream**: 25-30 FPS
- **Detection Speed**: Every 1 second
- **Latency**: <100ms
- **Memory Usage**: ~500MB
- **CPU Usage**: 15-25% (varies by system)

### Optimization Tips
- Use GPU if available (CUDA)
- Lower video resolution for better performance
- Reduce detection frequency for slower systems
- Close unnecessary applications

---

**🌊 Happy Underwater Species Detection!** 🐟🔍✨

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pygame Audio Documentation](https://www.pygame.org/docs/)

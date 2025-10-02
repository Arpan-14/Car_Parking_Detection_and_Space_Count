# 🚗 Car Parking Detection and Space Count System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.2-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.1.3-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A real-time car parking space detection system using deep learning that automatically counts available parking spaces from video feeds. The system uses computer vision and neural networks to classify parking spots as occupied or empty, providing live monitoring capabilities.

## 🌟 Features

- **Real-time Detection**: Live parking space monitoring from video feeds
- **Deep Learning**: CNN-based classification model for accurate detection
- **Web Interface**: User-friendly Flask web application with live video stream
- **Streamlit App**: Interactive Streamlit interface for image/video uploads
- **Space Counting**: Real-time count of available parking spaces
- **Visual Feedback**: Color-coded bounding boxes (Green: Available, Red: Occupied)
- **TensorFlow Lite**: Optimized model for faster inference
- **Multi-format Support**: Works with video files and live camera feeds

## 📋 Table of Contents

- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Training](#-training)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Demo

### Web Interface
The Flask web application provides:
- Live video stream with parking space detection
- Real-time count of available spaces
- Responsive Bootstrap UI
- REST API for integration

### Streamlit Interface
The Streamlit app offers:
- Upload and analyze parking lot images
- Process video files
- Interactive visualization
- Real-time results

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam or video file for testing

### Clone Repository
```bash
git clone https://github.com/Arpan-14/Car_Parking_Detection_and_Space_Count.git
cd Car_Parking_Detection_and_Space_Count
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Requirements
If you encounter any issues, install these packages individually:
```bash
pip install tensorflow==2.9.2
pip install opencv-python opencv-contrib-python
pip install flask==2.1.3
pip install streamlit
pip install numpy==1.24.4
pip install keras==2.9.0
pip install pillow
```

## 💻 Usage

### Method 1: Flask Web Application

1. **Start the Flask server:**
```bash
python main.py
```

2. **Access the web interface:**
   - Open your browser and go to: `http://localhost:5000`
   - View live parking detection with real-time space counting

3. **API Usage:**
   - GET `/` - Main interface
   - GET `/video_feed` - Live video stream
   - GET `/parking_count` - JSON response with available spaces

### Method 2: Streamlit Application

1. **Run the Streamlit app:**
```bash
streamlit run app.py
```

2. **Use the interface:**
   - Upload parking lot images or videos
   - View real-time detection results
   - Download processed outputs

### Method 3: Command Line Testing

1. **Test with sample data:**
```bash
python test.py
```

## 📁 Project Structure

```
Car_Parking_Detection_and_Space_Count/
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 📄 main.py                           # Flask web application
├── 📄 app.py                            # Streamlit application
├── 📄 test.py                           # Testing script
├── 📄 datacollection.py                 # Data collection utilities
├── 📄 convert_to_tflite.py             # Model conversion script
├── 📄 Car Parking System Using Deep Learning.ipynb  # Training notebook
├── 🤖 model_final.h5                    # Trained Keras model
├── 🤖 model_final.tflite               # TensorFlow Lite model
├── 📁 templates/
│   └── 📄 index.html                    # Web interface template
└── 📁 train_data/
    └── 📁 train_data/
        ├── 📁 train/
        │   ├── 📁 empty/                # Empty parking space images
        │   └── 📁 occupied/             # Occupied parking space images
        └── 📁 test/
            ├── 📁 empty/                # Test empty space images
            └── 📁 occupied/             # Test occupied space images
```

## 🧠 Model Architecture

### Neural Network Design
- **Input Shape**: 48x48x3 (RGB images)
- **Architecture**: Convolutional Neural Network (CNN)
- **Layers**: 
  - Multiple Conv2D layers with ReLU activation
  - MaxPooling2D for downsampling
  - Dropout layers for regularization
  - Dense layers for classification
- **Output**: Binary classification (Empty/Occupied)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

### Model Performance
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Model Size**: 112.35 MB (H5), Optimized TFLite version available
- **Inference Time**: ~10ms per detection

## 📊 Dataset

### Data Collection
- **Total Images**: 1000+ parking space images
- **Categories**: 
  - Empty parking spaces
  - Occupied parking spaces
- **Image Size**: 130x65 pixels (cropped ROIs)
- **Training Split**: 80% training, 20% testing

### Data Structure
```
train_data/
├── train/
│   ├── empty/     # 400+ empty space images
│   └── occupied/  # 400+ occupied space images
└── test/
    ├── empty/     # 100+ empty space images
    └── occupied/  # 100+ occupied space images
```

## 🎯 Training

### Training Process
1. **Data Preprocessing**: 
   - Image normalization (0-1 range)
   - Resizing to 48x48 pixels
   - Data augmentation

2. **Model Training**:
   ```bash
   # Run the Jupyter notebook
   jupyter notebook "Car Parking System Using Deep Learning.ipynb"
   ```

3. **Model Conversion**:
   ```bash
   python convert_to_tflite.py
   ```

### Training Parameters
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Validation Split**: 20%

## 🔗 API Endpoints

### Flask API

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | Main web interface | HTML page |
| `/video_feed` | GET | Live video stream | Video stream |
| `/parking_count` | GET | Get available spaces | JSON |

### Example API Response
```json
{
    "available_spaces": 15,
    "total_spaces": 20,
    "occupancy_rate": 25.0,
    "timestamp": "2025-10-02T10:30:00Z"
}
```

## ⚙️ Configuration

### Model Paths
Update file paths in the scripts according to your system:
```python
# main.py
model = load_model('path/to/model_final.h5')

# app.py  
interpreter = tf.lite.Interpreter(model_path="path/to/model_final.tflite")
```

### Video Source
Change video source in `main.py`:
```python
# For webcam
cap = cv2.VideoCapture(0)

# For video file
cap = cv2.VideoCapture('path/to/video.mp4')
```

### Parking Positions
The system uses `carposition.pkl` to define parking space coordinates. To create new positions:
1. Run `datacollection.py`
2. Click on corners of each parking space
3. Save the positions

## 🚀 Deployment

### Local Deployment
```bash
# Flask app
python main.py

# Streamlit app  
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]
```

### Cloud Deployment
- **Heroku**: Use `Procfile` for web deployment
- **AWS**: Deploy using EC2 or Lambda
- **Google Cloud**: Use App Engine or Cloud Run

## 🧪 Testing

### Unit Tests
```bash
python test.py
```

### Performance Testing
- Test with different video resolutions
- Measure inference time
- Validate accuracy on test dataset

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit changes**:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure backward compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Arpan**
- GitHub: [@Arpan-14](https://github.com/Arpan-14)
- Project: [Car Parking Detection and Space Count](https://github.com/Arpan-14/Car_Parking_Detection_and_Space_Count)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Flask and Streamlit for web frameworks
- Contributors and testers

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Issues**: [GitHub Issues](https://github.com/Arpan-14/Car_Parking_Detection_and_Space_Count/issues)
2. **Create New Issue**: Describe your problem with system details
3. **Discussion**: Use GitHub Discussions for general questions

## 🔮 Future Enhancements

- [ ] Real-time database integration
- [ ] Mobile app development
- [ ] Multi-camera support
- [ ] Advanced analytics dashboard
- [ ] Integration with parking management systems
- [ ] Support for different parking lot layouts
- [ ] Weather condition adaptation
- [ ] Night vision capabilities

---

⭐ **If you found this project helpful, please give it a star!** ⭐

![Parking Detection Demo](https://via.placeholder.com/800x400/0066cc/ffffff?text=Parking+Detection+Demo)
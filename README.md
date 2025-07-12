# Smart Diagnosis: AI-Powered Medical Report Analysis

⚕️ **Revolutionizing Healthcare with Intelligent Diagnostics**

Smart Diagnosis is an innovative, browser-based platform that leverages Artificial Intelligence (AI) and Natural Language Processing (NLP) to simplify and accelerate medical diagnostics. Our system empowers users to upload medical reports, get instant AI-powered analysis, and interact with a smart chatbot that explains results in easy-to-understand language.

## 🌟 Features

### 🔬 Multi-Modal Medical Analysis
- **Cancer Detection**: Breast cancer, brain tumor, and axial scan analysis using custom YOLO models
- **Fracture Detection**: Palm fracture detection from X-ray images with confidence scoring
- **Vitiligo Analysis**: Advanced skin condition detection with both upload and live camera options
- **Medical Chatbot**: AI-powered assistant for medical questions and comprehensive document analysis

### 🤖 AI-Powered Technologies
- **YOLO Models**: Custom-trained object detection models for each medical condition
- **Google Gemini AI**: Advanced language model for medical report interpretation and patient-friendly explanations
- **Real-time Analysis**: Live camera feed analysis for immediate vitiligo detection
- **Multi-format Support**: Images (JPG, PNG), PDFs, and text documents with OCR capabilities

### 💬 User-Friendly Interface
- **Conversational AI**: Explains complex medical terms in simple, understandable language
- **Visual Results**: Annotated images showing detected areas of concern with bounding boxes
- **Dual Reporting**: Both technical analysis for professionals and simplified explanations for patients
- **Interactive Navigation**: Easy-to-use sidebar navigation between different analysis modules

## 🎯 Live Demo Features

✅ **Upload Medical Reports**: Support for various medical scan formats  
📊 **AI-Processed Results**: Categorized analysis delivered in seconds  
🗨️ **Conversational Explanations**: Instant, easy-to-understand explanations via chatbot  
🧍‍♂️🔬 **NEW: Live Vitiligo Detection**: Real-time camera-based skin condition analysis  
📷 **Webcam Integration**: Use your device camera for instant skin analysis  
🧬 **Area Highlighting**: AI identifies and highlights affected areas with confidence scores  
🗣️ **Instant Recommendations**: System explains findings and suggests potential next steps

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/siddhardha-mns/AI-DIAGNOSIS-CHATBOT.git
   cd AI-DIAGNOSIS-CHATBOT
   ```

2. **Install required dependencies**
   ```bash
   pip install streamlit
   pip install Pillow
   pip install ultralytics
   pip install google-generativeai
   pip install opencv-python
   pip install PyPDF2
   ```

3. **Set up Google Gemini API**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace the API key in the code:
     ```python
     API_KEY = "YOUR_GEMINI_API_KEY_HERE"
     ```

4. **Download AI Models**
   Ensure you have the following YOLO model files in your project directory:
   - `breastcancer.pt` - Breast cancer detection model
   - `braintumor.pt` - Brain tumor detection model
   - `axialmri.pt` - Axial MRI analysis model
   - `fracture.pt` - Fracture detection model
   - `vitiligo.pt` - Vitiligo detection model

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📋 Usage Guide

### 1. Home Page
- Overview of the platform's capabilities
- Information about the team and technology
- Benefits and impact of the system

### 2. Cancer Analysis
- Select cancer type (Breast Cancer, Brain Tumor, Axial)
- Upload medical scan images
- Get AI-powered detection results
- View technical analysis and simplified explanations

### 3. Fracture Analysis
- Upload X-ray images for palm fracture detection
- Receive detailed analysis with confidence levels
- Get patient-friendly explanations of findings

### 4. Vitiligo Analysis
- **Upload Mode**: Analyze uploaded skin images
- **Live Camera Mode**: Real-time skin analysis using webcam
- Capture and analyze specific frames
- Get comprehensive skin condition reports

### 5. Medical Chatbot
- **Text Chat**: Ask medical questions and get AI responses
- **Document Analysis**: Upload medical reports for AI interpretation
- Support for images, PDFs, and text documents
- Conversation history tracking

## 🛠️ Technical Architecture

### Core Technologies
- **Frontend**: Streamlit for web interface
- **AI Models**: YOLOv8 for object detection
- **NLP**: Google Gemini AI for text analysis
- **Computer Vision**: OpenCV for image processing
- **Document Processing**: PyPDF2 for PDF analysis

### Model Pipeline
1. **Image Preprocessing**: Resize and normalize uploaded images
2. **Object Detection**: YOLO models identify areas of concern
3. **Confidence Analysis**: Extract detection confidence scores
4. **AI Interpretation**: Gemini AI generates human-readable explanations
5. **Results Visualization**: Annotated images with bounding boxes

## 📁 Project Structure

```
smart-diagnosis/
├── app.py                 # Main Streamlit application
├── models/
│   ├── breastcancer.pt    # Breast cancer detection model
│   ├── braintumor.pt      # Brain tumor detection model
│   ├── axialmri.pt        # Axial MRI analysis model
│   ├── fracture.pt        # Fracture detection model
│   └── vitiligo.pt        # Vitiligo detection model
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### API Keys
Replace the placeholder API key with your actual Google Gemini API key:
```python
API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
```

### Model Paths
Ensure all model files are in the same directory as your main application file.

## 🚨 Important Disclaimers

- **Medical Advice**: This application is for demonstration and educational purposes only
- **Professional Consultation**: Always consult qualified healthcare professionals for medical diagnosis
- **AI Limitations**: AI analysis should not replace professional medical evaluation
- **Data Privacy**: Medical data should be handled according to healthcare privacy regulations

## 🎯 Use Cases

### For Patients
- **Quick Screening**: Initial analysis of medical scans
- **Report Understanding**: Simplified explanations of complex medical reports
- **Health Monitoring**: Track skin conditions over time

### For Healthcare Providers
- **Second Opinion**: AI-assisted diagnosis support
- **Patient Education**: Visual explanations for patient consultations
- **Workflow Efficiency**: Rapid preliminary analysis of medical scans
- **Documentation**: Automated report generation with technical details

### For Researchers & Students
- **Educational Tool**: Learn about medical imaging and AI applications
- **Research Platform**: Analyze patterns in medical data
- **Technology Demonstration**: Showcase AI capabilities in healthcare

## 🏆 Team HC-16

**Meet the Innovators:**
- **MNS Siddhardha** - Matrusri Engineering College (Project Lead)
- **Kalwa Sanketh** - Vignan Institute of Science and Technology
- **K Sri Charan Karthik** - Vignan Institute of Science and Technology  
- **M Shiva** - Vignan Institute of Science and Technology

## 🚀 Future Roadmap

### Upcoming Features
- 📲 **Mobile App**: Native iOS and Android applications
- 📡 **Real-Time Health Monitoring**: Continuous health tracking capabilities
- 🏥 **Hospital Integration**: EHR/EMR system compatibility
- 🌍 **Multi-language Support**: Chatbot support for multiple languages
- 🩺 **Telemedicine Integration**: Direct consultation booking
- 🔬 **Advanced AI Models**: More specialized detection models

### Technical Improvements
- **Cloud Deployment**: Scalable cloud infrastructure
- **API Development**: RESTful API for third-party integrations
- **Performance Optimization**: Faster model inference
- **Security Enhancement**: HIPAA-compliant data handling
- **Mobile Optimization**: Responsive design for all devices

## 📊 Model Performance

### Detection Accuracy
- **Breast Cancer Detection**: Trained on comprehensive mammography datasets
- **Brain Tumor Analysis**: MRI and CT scan specialized models
- **Fracture Detection**: X-ray image analysis with high precision
- **Vitiligo Detection**: Skin condition identification with real-time capability

### AI Capabilities
- **Multi-modal Analysis**: Combined image and text processing
- **Natural Language Processing**: Medical terminology simplification
- **Computer Vision**: Advanced object detection and classification
- **Real-time Processing**: Live camera feed analysis

## 🛡️ Security & Privacy

- **Data Protection**: No medical data stored permanently
- **Local Processing**: Image analysis performed locally when possible
- **API Security**: Encrypted communication with external services
- **Privacy Compliance**: Designed with healthcare privacy standards in mind

## 📞 Support & Contribution

### Getting Help
- **Issues**: Report bugs or feature requests via GitHub Issues
- **Documentation**: Comprehensive guides and API documentation
- **Community**: Join discussions and get support from the community

### Contributing
We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### License
This project is open-source. Please check the LICENSE file for details.

## 📈 Impact & Benefits

### For Patients
- ⏱️ **Faster Diagnoses**: Reduce waiting time for medical analysis
- 📘 **Health Literacy**: Better understanding of medical conditions
- 💰 **Cost-Effective**: Reduce unnecessary medical visits
- 🏠 **Home Monitoring**: Early detection from home

### For Healthcare System
- 🏥 **Efficiency**: Streamline diagnostic workflows
- 🌐 **Accessibility**: Reach underserved communities
- 📊 **Data Insights**: Valuable health analytics
- 🌱 **Sustainability**: Reduce paper usage and optimize resources

---

## 🌐 Repository Information

**GitHub Repository**: [AI-DIAGNOSIS-CHATBOT](https://github.com/siddhardha-mns/AI-DIAGNOSIS-CHATBOT)

**Project Status**: Active Development  
**Latest Update**: July 2025  
**Version**: 1.0.0

---

*Smart Diagnosis - Empowering Healthcare Through AI Innovation*

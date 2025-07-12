import streamlit as st
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
import cv2
import time
from io import BytesIO

# --- Setup Gemini AI function that can be reused across pages ---
def setup_gemini():
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"  
    genai.configure(api_key=API_KEY) 
    model = genai.GenerativeModel("gemini-1.5-flash")  # Using Flash model
    return model

# --- Model Loader Function ---
def load_model_for_cancer(cancer_type):
    if cancer_type == "Breast Cancer":
        return YOLO("breastcancer.pt")
    elif cancer_type == "Brain Tumor":
        return YOLO("braintumor.pt")
    elif cancer_type == "Axial":
        return YOLO("axialmri.pt")
    else:
        return None

def main_page():
    st.title("⚕️ Smart Diagnosis: Harnessing AI for Medical Report Analysis")
    st.subheader("Revolutionizing Healthcare with Intelligent Diagnostics")
    st.markdown("""
  

    **🚨 The Challenge**

    Understanding medical reports like ECGs, MRIs, CT scans, and X-rays can be overwhelming. Delays in expert interpretation, combined with complex medical language, leave patients confused and anxious about their health.

    **💡 Our Solution**

    **Smart Diagnosis** is an innovative, browser-based platform that leverages **Artificial Intelligence (AI)** and **Natural Language Processing (NLP)** to simplify and accelerate medical diagnostics.

    **With our system, users can:**

    📤 Upload medical reports
    🧠 Get instant AI-powered analysis and classification (e.g., “Normal ECG”, “Abnormal MRI”)
    💬 Interact with a smart chatbot that explains results in easy-to-understand language

    **🌟 What Makes Us Unique**

    🔄 **All-in-One Report Support:** Handles ECGs, MRIs, CT scans, X-rays, and more
    🤖 **Automated Intelligence:** No need to wait for an expert — AI gives fast, reliable insights
    🗣️ **Conversational Clarity:** Chatbot breaks down medical jargon into plain speech
    👨‍👩‍👧‍👦 **Empowering for Patients:** Makes health data accessible and understandable
    🌐 **Web-Based and Easy:** No installation needed — just open your browser and start


    **🧪 Live Demo – What You Can Experience**

    ✅ Upload various types of medical reports
    📊 Receive categorized, AI-processed results in seconds
    🗨️ Get instant, conversational explanations through our chatbot
    🧍‍♂️🔬 **NEW: Live Vitiligo Detection via Camera**

    Now featuring real-time camera-based detection for vitiligo — a condition causing white patches on the skin.

    📷 Use your webcam or phone camera for instant skin analysis
    🧬 AI identifies and highlights vitiligo-affected areas
    🗣️ The system explains what it sees and recommends potential next steps

    **Early detection from the comfort of your home.**

                
    **🔧 Benefits & Impact**

    ⏱️ **Faster Diagnoses** – No more waiting in uncertainty
    👨‍⚕️ **Support for Doctors** – Acts as a second opinion in clinical settings
    📘 **Health Literacy** – Makes patients more informed and confident
    🏥 **Versatile Use** – Clinics, telemedicine, emergency care, and home monitoring


    **🌍 Built to Scale & Sustain**

    ☁️ **Cloud-Based Infrastructure** – Grows with demand
    🌐 **Supports Multi-Language Use** – Perfect for global and rural reach
    📱 **Mobile-Ready** – Future versions will support smartphones and wearables
    🌱 **Eco-Friendly** – Digital reports reduce paper use and unnecessary tests
    💰 **Cost-Effective** – Ideal for under-resourced areas, with options for free and premium services


    **🚀 What’s Coming Next**

    📲 Mobile App
    📡 Real-Time Health Monitoring
    🏥 Integration with Hospital Systems (EHR/EMR)
    🌍 Multi-language Chatbot Support
    🩺 Telemedicine Integration


    **👨‍💻 Meet the Innovators – Team HC-16**

    MNS Siddhardha – Matrusri Engineering College
    Kalwa Sanketh – Vignan Institute of Science and Technology
    K Sri Charan Karthik – Vignan Institute of Science and Technology
    M Shiva – Vignan Institute of Science and Technology
    """)

# --- Cancer Analysis Page ---
def cancer_analysis_page():
    st.title("Cancer Scan Analysis")
    st.subheader("Select Cancer Type for AI-Based Analysis")

    # Initialize Gemini AI model for summarization
    gemini_model = setup_gemini()

    cancer_type = st.selectbox("Select Cancer Type:", ["None", "Breast Cancer", "Brain Tumor", "Axial"])

    if 'cancer_submit' not in st.session_state:
        st.session_state.cancer_submit = False
        st.session_state.selected_cancer = None

    if cancer_type != "None":
        if st.button("Submit"):
            st.session_state.cancer_submit = True
            st.session_state.selected_cancer = cancer_type

    if st.session_state.cancer_submit and st.session_state.selected_cancer == cancer_type:
        st.subheader(f"{cancer_type} Information & AI Analysis")
        st.info(f"You selected {cancer_type}. Upload an image to run the specialized detection model.")

        # Show detailed information based on cancer type
        if cancer_type == "Brain Tumor":
            st.markdown("""
            ### 🧠 Brain Tumour (MRI/CT Scan Analysis)
            
            Brain tumours can be either benign (non-cancerous) or malignant (cancerous). 
            Our platform uses advanced AI models to detect abnormalities in brain scans like:
            
            - Tumour size, shape, and location
            - Signs of pressure or swelling in the brain
            - Comparison against healthy brain tissue
            
            **Why it matters**: Early detection helps in timely treatment and better recovery chances.
            """)
        elif cancer_type == "Axial":
            st.markdown("""
            ### 📸 Axial (Cross-Sectional Imaging)
            
            Axial scans are cross-sectional images taken along the horizontal plane of the body, typically seen in:
            - MRI, CT scans
            - Brain, chest, and abdomen evaluations
            
            Our system helps you understand axial views by providing:
            - Easy-to-read annotations
            - Highlighted problem areas
            - Descriptive summaries of abnormalities
            """)
        elif cancer_type == "Breast Cancer":
            st.markdown("""
            ### 🎗️ Breast Cancer (Mammogram/MRI Analysis)
            
            Breast cancer diagnosis involves identifying:
            - Lumps, masses, or unusual density in breast tissue
            - Calcifications or changes over time
            
            Smart Diagnosis offers:
            - AI-based identification of suspicious areas
            - Classification into benign/malignant categories
            - Supportive chatbot explaining risk levels and next steps
            
            Empowering early action and awareness for better outcomes.
            """)

        uploaded_file = st.file_uploader(f"Upload an image for {cancer_type} detection...", type=["jpg", "jpeg", "png"], key=f"{cancer_type}_uploader")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)

            # Load appropriate model and run detection
            model = load_model_for_cancer(cancer_type)
            if model:
                with st.spinner(f"Running AI detection using {cancer_type} model..."):
                    results = model(image)

                st.success("Detection complete!")
                res_image = results[0].plot()
                st.image(res_image, caption='Detected Findings', use_container_width=True)
                
                # Get technical results from model
                detection_results = results[0].boxes
                num_detections = len(detection_results)
                
                # Prepare technical output
                technical_output = f"Detection found {num_detections} potential areas of concern in the {cancer_type} scan."
                
                if num_detections > 0:
                    technical_output += "\n\nTechnical details:"
                    for i, box in enumerate(detection_results):
                        confidence = float(box.conf[0])
                        technical_output += f"\n- Detection {i+1}: Confidence level: {confidence:.2%}"
                        if hasattr(box, 'cls') and len(box.cls) > 0:
                            class_id = int(box.cls[0])
                            technical_output += f", Class: {class_id}"
                
                st.subheader("Technical Analysis")
                st.text(technical_output)
                
                # Generate simplified explanation with Gemini AI
                with st.spinner("Generating simplified explanation..."):
                    try:
                        # Convert image to bytes for processing by Gemini
                        import io
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format)
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Create a prompt for Gemini to analyze both the image and detection results
                        prompt = f"""
                        This is a medical scan for {cancer_type}. 
                        
                        Technical analysis results: {technical_output}
                        
                        Please provide a simplified explanation of what these findings might mean in everyday language 
                        that a patient could understand. Explain what the patient should know and potential next steps,
                        while emphasizing that this is just an AI analysis and proper medical consultation is required.
                        Keep your explanation concise and reassuring.
                        """
                        
                        # Get simplified explanation from Gemini
                        response = gemini_model.generate_content([
                            prompt,
                            {"mime_type": f"image/{image.format.lower()}", "data": img_bytes}
                        ])
                        
                        st.subheader("Simplified Explanation")
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Error generating simplified explanation: {e}")
            else:
                st.error("Model not available for the selected cancer type.")

        st.warning("This information is for demonstration purposes only and should not replace medical advice.")

# --- Fracture Analysis Page ---
def fracture_analysis_page():
    st.title("Fracture Scan Analysis")
    st.subheader("Select Fracture Type for Further Information")

    # Initialize Gemini AI model for summarization
    gemini_model = setup_gemini()

    fracture_type = st.selectbox("Select Fracture Type:", ["None", "Palm Fracture"])

    if 'palm_submit' not in st.session_state:
        st.session_state.palm_submit = False

    if fracture_type == "Palm Fracture":
        if st.button("Submit"):
            st.session_state.palm_submit = True

        if st.session_state.palm_submit:
            st.subheader("Palm Fracture Information & AI Detection")
            st.info("You selected Palm Fracture. This section includes education and image-based detection.")

            st.markdown("""
            ### About Palm Fracture
            - Involves break in hand bones
            - Common from falls or trauma
            - Requires X-ray for diagnosis
            """)

            uploaded_file = st.file_uploader("Upload a hand X-ray image...", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)

                with st.spinner("Running fracture detection..."):
                    results = YOLO("fracture.pt")(image)

                st.success("Detection complete!")
                res_image = results[0].plot()
                st.image(res_image, caption='Detected Fractures', use_container_width=True)
                
                # Get technical results from model
                detection_results = results[0].boxes
                num_detections = len(detection_results)
                
                # Prepare technical output
                technical_output = f"Detection found {num_detections} potential fracture areas in the scan."
                
                if num_detections > 0:
                    technical_output += "\n\nTechnical details:"
                    for i, box in enumerate(detection_results):
                        confidence = float(box.conf[0])
                        technical_output += f"\n- Fracture {i+1}: Confidence level: {confidence:.2%}"
                        if hasattr(box, 'cls') and len(box.cls) > 0:
                            class_id = int(box.cls[0])
                            technical_output += f", Type: {class_id}"
                
                st.subheader("Technical Analysis")
                st.text(technical_output)
                
                # Generate simplified explanation with Gemini AI
                with st.spinner("Generating simplified explanation..."):
                    try:
                        # Convert image to bytes for processing by Gemini
                        import io
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format)
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Create a prompt for Gemini to analyze both the image and detection results
                        prompt = f"""
                        This is an X-ray scan for a potential palm fracture. 
                        
                        Technical analysis results: {technical_output}
                        
                        Please provide a simplified explanation of what these findings might mean in everyday language 
                        that a patient could understand. Explain what the patient might experience, potential recovery 
                        timeline, and basic care guidelines in simple terms. Emphasize that this is just an AI analysis 
                        and proper medical consultation is required. Keep your explanation concise and reassuring.
                        """
                        
                        # Get simplified explanation from Gemini
                        response = gemini_model.generate_content([
                            prompt,
                            {"mime_type": f"image/{image.format.lower()}", "data": img_bytes}
                        ])
                        
                        st.subheader("Simplified Explanation")
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"Error generating simplified explanation: {e}")

            st.warning("This analysis is for demonstration only and should not replace medical advice.")

# --- NEW: Vitiligo Analysis Page ---
def vitiligo_analysis_page():
    st.title("Vitiligo Scan Analysis")
    st.subheader("AI-Powered Vitiligo Detection & Analysis")
    
    # Initialize Gemini AI model for summarization
    gemini_model = setup_gemini()
    
    # Create tabs for upload and live camera
    tab1, tab2 = st.tabs(["Upload Image", "Live Camera Analysis"])
    
    with tab1:
        st.markdown("""
        ### About Vitiligo
        - A condition causing loss of skin color in patches
        - Results from lack of melanin production
        - Affects any area of skin
        - Not contagious or life-threatening
        - Can be managed with various treatments
        """)
        
        uploaded_file = st.file_uploader("Upload an image for vitiligo detection...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Load vitiligo model and run detection
            with st.spinner("Running AI detection using vitiligo model..."):
                try:
                    vitiligo_model = YOLO("vitiligo.pt")
                    results = vitiligo_model(image)
                    
                    st.success("Detection complete!")
                    res_image = results[0].plot()
                    st.image(res_image, caption='Detected Vitiligo Areas', use_container_width=True)
                    
                    # Get technical results from model
                    detection_results = results[0].boxes
                    num_detections = len(detection_results)
                    
                    # Prepare technical output
                    technical_output = f"Detection found {num_detections} potential vitiligo affected areas in the image."
                    
                    if num_detections > 0:
                        technical_output += "\n\nTechnical details:"
                        for i, box in enumerate(detection_results):
                            confidence = float(box.conf[0])
                            technical_output += f"\n- Area {i+1}: Confidence level: {confidence:.2%}"
                            if hasattr(box, 'cls') and len(box.cls) > 0:
                                class_id = int(box.cls[0])
                                technical_output += f", Type: {class_id}"
                    
                    st.subheader("Technical Analysis")
                    st.text(technical_output)
                    
                    # Generate simplified explanation with Gemini AI
                    with st.spinner("Generating simplified explanation..."):
                        try:
                            # Convert image to bytes for processing by Gemini
                            img_byte_arr = BytesIO()
                            image.save(img_byte_arr, format=image.format)
                            img_bytes = img_byte_arr.getvalue()
                            
                            # Create a prompt for Gemini to analyze both the image and detection results
                            prompt = f"""
                            This is a skin image for vitiligo analysis.
                            
                            Technical analysis results: {technical_output}
                            
                            Please provide a simplified explanation of what these findings might mean in everyday language
                            that a patient could understand. Include:
                            1. What the detection shows about potential vitiligo areas
                            2. General information about what vitiligo is
                            3. Potential treatment approaches
                            4. Lifestyle considerations
                            
                            Keep your explanation concise and reassuring, emphasizing that this is just an AI analysis
                            and proper dermatological consultation is required.
                            """
                            
                            # Get simplified explanation from Gemini
                            response = gemini_model.generate_content([
                                prompt,
                                {"mime_type": f"image/{image.format.lower()}", "data": img_bytes}
                            ])
                            
                            st.subheader("Simplified Explanation")
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"Error generating simplified explanation: {e}")
                except Exception as e:
                    st.error(f"Error during detection: {e}")
            
    with tab2:
        st.markdown("### Live Camera Vitiligo Detection")
        st.info("This feature uses your webcam to analyze skin for potential vitiligo areas in real-time.")
        
        # Initialize session state for camera
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
        
        # Initialize session state for captured frame
        if 'captured_frame' not in st.session_state:
            st.session_state.captured_frame = None
        
        # Button to start/stop camera
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button('Start Camera' if not st.session_state.camera_on else 'Stop Camera')
            if start_button:
                st.session_state.camera_on = not st.session_state.camera_on
                # Clear captured frame when stopping camera
                if not st.session_state.camera_on:
                    st.session_state.captured_frame = None
        
        with col2:
            # Add capture button (only enabled when camera is on)
            capture_button = st.button('Capture Frame', disabled=not st.session_state.camera_on)
        
        # Create placeholder for camera feed and captured frame analysis
        video_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Create placeholders for captured frame and analysis
        captured_img_placeholder = st.empty()
        captured_analysis_placeholder = st.empty()
        
        # Load vitiligo model for camera feed
        vitiligo_model = YOLO("vitiligo.pt")
        
        # Handle camera operations
        if st.session_state.camera_on:
            try:
                # Set up camera capture
                cap = cv2.VideoCapture(0)  # Using default camera (0)
                
                # Check if camera opened successfully
                if not cap.isOpened():
                    st.error("Error: Could not open camera.")
                    st.session_state.camera_on = False
                else:
                    info_placeholder.info("Camera started. Processing live feed...")
                    
                    # Main camera loop
                    while st.session_state.camera_on:
                        # Read a frame from the camera
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("Failed to capture image")
                            break
                        
                        # Convert frame from BGR (OpenCV format) to RGB (PIL format)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # If capture button is pressed, store the current frame
                        if capture_button:
                            st.session_state.captured_frame = frame_rgb.copy()
                            info_placeholder.success("Frame captured! Analyzing...")
                            break
                        
                        # Run detection on the frame
                        results = vitiligo_model(frame_rgb)
                        
                        # Get detection results image with bounding boxes
                        res_image = results[0].plot()
                        
                        # Display the processed frame
                        video_placeholder.image(res_image, caption='Live Detection', use_container_width=True)
                        
                        # Get detection stats
                        detection_results = results[0].boxes
                        num_detections = len(detection_results)
                        
                        if num_detections > 0:
                            stats_text = f"Detected {num_detections} potential vitiligo areas"
                            confidences = [float(box.conf[0]) for box in detection_results]
                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                            stats_text += f" | Average confidence: {avg_confidence:.2%}"
                            info_placeholder.info(stats_text)
                        else:
                            info_placeholder.info("No vitiligo areas detected in current frame")
                        
                        # Add a short delay to control frame rate
                        time.sleep(0.1)
                    
                    # Release the camera when stopped
                    cap.release()
                    if not capture_button:
                        info_placeholder.info("Camera stopped")
                
            except Exception as e:
                st.error(f"Camera error: {e}")
                st.session_state.camera_on = False
        
        # Process captured frame if it exists
        if st.session_state.captured_frame is not None:
            # Create PIL image from captured frame
            captured_img = Image.fromarray(st.session_state.captured_frame)
            
            # Show captured frame
            captured_img_placeholder.image(captured_img, caption='Captured Image', use_container_width=True)
            
            # Run vitiligo detection on the captured frame
            with captured_analysis_placeholder.container():
                st.subheader("Vitiligo Analysis of Captured Frame")
                
                with st.spinner("Analyzing captured frame..."):
                    # Run detection on the captured frame
                    results = vitiligo_model(st.session_state.captured_frame)
                    
                    # Display detection results
                    res_image = results[0].plot()
                    st.image(res_image, caption='Detected Vitiligo Areas', use_container_width=True)
                    
                    # Get technical results from model
                    detection_results = results[0].boxes
                    num_detections = len(detection_results)
                    
                    # Prepare technical output
                    technical_output = f"Detection found {num_detections} potential vitiligo affected areas in the image."
                    
                    if num_detections > 0:
                        technical_output += "\n\nTechnical details:"
                        for i, box in enumerate(detection_results):
                            confidence = float(box.conf[0])
                            technical_output += f"\n- Area {i+1}: Confidence level: {confidence:.2%}"
                            if hasattr(box, 'cls') and len(box.cls) > 0:
                                class_id = int(box.cls[0])
                                technical_output += f", Type: {class_id}"
                    
                    st.subheader("Technical Analysis")
                    st.text(technical_output)
                    
                    # Generate simplified explanation with Gemini AI
                    with st.spinner("Generating simplified explanation..."):
                        try:
                            # Convert image to bytes for processing by Gemini
                            img_byte_arr = BytesIO()
                            captured_img.save(img_byte_arr, format='JPEG')
                            img_bytes = img_byte_arr.getvalue()
                            
                            # Create a prompt for Gemini to analyze both the image and detection results
                            prompt = f"""
                            This is a skin image for vitiligo analysis.
                            
                            Technical analysis results: {technical_output}
                            
                            Please provide a simplified explanation of what these findings might mean in everyday language
                            that a patient could understand. Include:
                            1. What the detection shows about potential vitiligo areas
                            2. General information about what vitiligo is
                            3. Potential treatment approaches
                            4. Lifestyle considerations
                            
                            Keep your explanation concise and reassuring, emphasizing that this is just an AI analysis
                            and proper dermatological consultation is required.
                            """
                            
                            # Get simplified explanation from Gemini
                            response = gemini_model.generate_content([
                                prompt,
                                {"mime_type": "image/jpeg", "data": img_bytes}
                            ])
                            
                            st.subheader("Simplified Explanation")
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"Error generating simplified explanation: {e}")
            
                # Add a button to clear the captured frame
                if st.button("Clear Captured Frame"):
                    st.session_state.captured_frame = None
                    st.experimental_rerun()
        
        st.warning("This live analysis is for demonstration only. Please consult a dermatologist for proper diagnosis.")

# --- Chatbot: Report Analyzer Page (Google Gemini AI) ---
def chatbot_page():
    st.title("Medical Chatbot")
    st.subheader("Interact with our AI-powered Medical Assistant")

    # Setup Gemini Model
    model = setup_gemini()

    # If session state for conversation doesn't exist, initialize it
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Create tabs for text chat and document analysis
    tab1, tab2 = st.tabs(["Chat with Assistant", "Upload & Analyze Documents"])
    
    with tab1:
        st.markdown("### Ask any medical questions")
        
        # Input text area for user to ask the chatbot
        user_input = st.text_input("Ask the AI Assistant:", placeholder="Example: What are common symptoms of diabetes?")

        # On submit, send the user's input and generate a response
        if st.button("Send", key="send_text") and user_input:
            # Add user input to conversation history
            st.session_state.conversation.append(f"You: {user_input}")
            
            with st.spinner("AI is processing your query..."):
                try:
                    # Generate AI response using the Google Gemini model
                    response = model.generate_content(user_input)
                    ai_reply = response.text
                    
                    # Add AI response to conversation history
                    st.session_state.conversation.append(f"AI: {ai_reply}")

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.session_state.conversation.append("AI: Sorry, I encountered an error processing your request.")
    
    with tab2:
        st.markdown("### Upload Medical Documents for AI Analysis")
        st.markdown("Upload medical reports, scans, or lab results for the AI to analyze and explain in simple terms.")
        
        uploaded_file = st.file_uploader("Upload a medical document or report", type=["pdf", "txt", "jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the file details
            file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size} bytes"}
            st.write(file_details)
            
            # Process different file types
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                # Convert image to bytes for processing
                import io
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_bytes = img_byte_arr.getvalue()
                
                with st.spinner("Analyzing image..."):
                    try:
                        # Process image with Gemini multimodal capabilities
                        vision_model = genai.GenerativeModel('gemini-1.5-flash')
                        response = vision_model.generate_content([
                            "Analyze this medical image and explain the findings in simple terms that a non-medical professional would understand. Highlight any notable observations:",
                            {"mime_type": f"image/{image.format.lower()}", "data": img_bytes}
                        ])
                        
                        st.subheader("AI Analysis:")
                        st.write(response.text)
                        
                        # Add a simpler summary section
                        st.subheader("Simple Summary:")
                        summary_prompt = "Based on the above analysis, give me a 2-3 sentence summary in very simple language that anyone could understand:"
                        summary = vision_model.generate_content(summary_prompt + response.text)
                        st.info(summary.text)
                        
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
                
            elif uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.txt'):
                # Read text content from uploaded file
                if uploaded_file.type == "application/pdf":
                    try:
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                    except ImportError:
                        st.error("PDF processing library not available. Please install PyPDF2.")
                        text_content = "Unable to extract PDF content."
                else:  # Text file
                    text_content = uploaded_file.getvalue().decode("utf-8")
                
                # Show a preview of the content
                with st.expander("Document Preview"):
                    st.text(text_content[:500] + "..." if len(text_content) > 500 else text_content)
                
                with st.spinner("Analyzing document..."):
                    try:
                        # Process document with Gemini
                        analysis_prompt = """
                        Analyze this medical document and provide the following:
                        1. Main findings or diagnoses in simple terms
                        2. Explanation of any medical terminology used
                        3. What this might mean for the patient (in non-technical language)
                        4. Any follow-up actions that might be needed
                        
                        Present this information in a way that someone without medical training can understand:
                        """
                        
                        response = model.generate_content(analysis_prompt + text_content)
                        
                        st.subheader("AI Analysis:")
                        st.write(response.text)
                        
                        # Add a simpler summary section
                        st.subheader("Simple Summary:")
                        summary_prompt = "Based on the above analysis, give me a 2-3 sentence summary of the key points in very simple language:"
                        summary = model.generate_content(summary_prompt + response.text)
                        st.info(summary.text)
                        
                    except Exception as e:
                        st.error(f"Error analyzing document: {e}")
    
    # Show the conversation history
    st.subheader("Conversation History")
    if st.session_state.conversation:
        for msg in st.session_state.conversation:
            if msg.startswith("You: "):
                st.markdown(f"**{msg}**")
            else:
                st.markdown(msg)
    else:
        st.markdown("_Your conversation will appear here._")
        
    
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.experimental_rerun()

# --- Sidebar Navigation ---
st.sidebar.title("Medicare Navigation")
page = st.sidebar.radio("Go to:", ("Home", "Cancer Analysis", "Fracture Analysis", "Vitiligo Analysis", "Medical Chatbot"))

if page == "Home":
    main_page()
elif page == "Cancer Analysis":
    cancer_analysis_page()
elif page == "Fracture Analysis":
    fracture_analysis_page()
elif page == "Vitiligo Analysis":
    vitiligo_analysis_page()
elif page == "Medical Chatbot":
    chatbot_page()

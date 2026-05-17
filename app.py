import os
import cv2 as cv
import tempfile
import logging
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

from PIL import Image
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Wildlife Surveillance",
    page_icon="🦁",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>

    .main {
        background-color: #050816;
        color: white;
    }

    .stApp {
        background-color: #050816;
    }

    h1, h2, h3 {
        color: white;
    }

    .block-container {
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# MODEL
# =========================

MODEL_DIR = './runs/detect/train/weights/best.pt'

model = YOLO(MODEL_DIR)

# =========================
# HAZARDOUS ANIMALS
# =========================

hazardous_animals = [
    "Lion",
    "Tiger",
    "Cheetah",
    "Jaguar",
    "Elephant"
]

# =========================
# FAST2SMS CONFIG
# =========================

FAST2SMS_API_KEY = "YmzkD6oOFGCRpclH4dhx7tSrsvnaT0VIUg1u5LwNKbAWf3qPBXmivJc10geyuSGlQLWKwBb3E4Ud7IrD"

ALERT_PHONE = "8618408819"

# =========================
# LOGGING
# =========================

logging.basicConfig(
    filename="./logs/log.log",
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# =========================
# HISTORY FILE
# =========================

history_file = "history.csv"

if not os.path.exists(history_file):

    df = pd.DataFrame(
        columns=[
            "Animal",
            "Type",
            "Confidence",
            "Location",
            "Time"
        ]
    )

    df.to_csv(history_file, index=False)

# =========================
# GET LOCATION
# =========================

def get_location():

    # Fixed Project Demo Location

    location_name = "Bidar, Karnataka"

    # Bidar Coordinates
    latitude = 17.9104
    longitude = 77.5199

    # Google Maps Link
    maps_link = (
        f"https://www.google.com/maps?q="
        f"{latitude},{longitude}"
    )

    return (
        location_name,
        maps_link
    )

# =========================
# SEND SMS
# =========================

def send_sms(animal_name):

    location_name, maps_link = get_location()

    message = (
        f"WARNING! Hazardous Animal Detected\n"
        f"Animal: {animal_name}\n"
        f"Location: {location_name}\n"
        f"Maps: {maps_link}"
    )

    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "sender_id": "FSTSMS",
        "message": message,
        "language": "english",
        "route": "q",
        "numbers": ALERT_PHONE
    }

    headers = {
        'authorization': FAST2SMS_API_KEY
    }

    try:

        requests.post(
            url,
            data=payload,
            headers=headers
        )

    except Exception as e:

        print("SMS ERROR:", e)

# =========================
# SAVE HISTORY
# =========================

def save_history(animal, animal_type, confidence):

    location_name, _ = get_location()

    df = pd.read_csv(history_file)

    new_row = {
        "Animal": animal,
        "Type": animal_type,
        "Confidence": confidence,
        "Location": location_name,
        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    df.loc[len(df)] = new_row

    df.to_csv(history_file, index=False)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    selected = option_menu(
        menu_title="AI Wildlife",
        options=[
            "Dashboard",
            "Detection",
            "History",
            "Analytics",
            "About"
        ],
        icons=[
            "house",
            "camera-video",
            "clock-history",
            "bar-chart",
            "info-circle"
        ],
        default_index=0,
    )

# =========================
# DASHBOARD
# =========================

if selected == "Dashboard":

    st.markdown(
        """
        <h1 style='text-align:center;
        color:white;
        font-size:52px;
        margin-top:20px;'>
        REAL-TIME HAZARDOUS ANIMAL DETECTION
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style='text-align:center;
        color:#94a3b8;
        margin-bottom:40px;'>
        AI-Based Wildlife Surveillance System Using YOLOv8
        </h3>
        """,
        unsafe_allow_html=True
    )

    # ABOUT PROJECT

    st.markdown(
        """
        <div style='
        background:#0B1120;
        padding:30px;
        border-radius:20px;
        border:1px solid #1e293b;
        margin-bottom:30px;
        '>

        <h2 style='color:white;'>
        About Project
        </h2>

        <p style='
        color:#cbd5e1;
        font-size:18px;
        line-height:1.8;
        '>

        This project is designed to detect hazardous and
        non-hazardous animals using the YOLOv8 deep learning model.

        The system processes wildlife images and videos in real time,
        identifies animal species, classifies dangerous animals,
        and sends alert notifications with location details.

        The application is developed using Python, Streamlit,
        OpenCV, and YOLOv8 for intelligent wildlife surveillance
        and monitoring.

        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # TECHNOLOGIES

    st.markdown(
        """
        <div style='
        background:#0B1120;
        padding:30px;
        border-radius:20px;
        border:1px solid #1e293b;
        margin-bottom:30px;
        '>

        <h2 style='color:white;'>
        Technologies Used
        </h2>

        <ul style='
        color:#cbd5e1;
        font-size:18px;
        line-height:2;
        '>

        <li>YOLOv8 Deep Learning Model</li>
        <li>Python Programming</li>
        <li>OpenCV Video Processing</li>
        <li>Streamlit Frontend</li>
        <li>Fast2SMS Alert System</li>
        <li>Google Maps Integration</li>
        <li>Real-Time Wildlife Detection</li>

        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )

    # FEATURES

    st.markdown(
        """
        <div style='
        background:#0B1120;
        padding:30px;
        border-radius:20px;
        border:1px solid #1e293b;
        margin-bottom:30px;
        '>

        <h2 style='color:white;'>
        Key Features
        </h2>

        <ul style='
        color:#cbd5e1;
        font-size:18px;
        line-height:2;
        '>

        <li>Hazardous Animal Detection</li>
        <li>Non-Hazardous Animal Classification</li>
        <li>Real-Time Image and Video Detection</li>
        <li>Automatic SMS Alert Generation</li>
        <li>Live Location and Google Maps Link</li>
        <li>Detection History Storage</li>
        <li>Analytics Dashboard</li>

        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )

    # LATEST DETECTION

    st.markdown(
        """
        <h2 style='color:white; margin-top:40px;'>
        Latest Detection
        </h2>
        """,
        unsafe_allow_html=True
    )

    try:

        df = pd.read_csv(history_file)

        if len(df) > 0:

            latest = df.iloc[-1]

            st.markdown(
                f"""
                <div style='
                background:#111827;
                padding:25px;
                border-radius:18px;
                border:1px solid #374151;
                '>

                <h3 style='color:#22c55e;'>
                Animal: {latest['Animal']}
                </h3>

                <p style='color:white; font-size:18px;'>
                Type: {latest['Type']}
                </p>

                <p style='color:white; font-size:18px;'>
                Confidence: {latest['Confidence']}%
                </p>

                <p style='color:white; font-size:18px;'>
                Location: {latest['Location']}
                </p>

                <p style='color:white; font-size:18px;'>
                Time: {latest['Time']}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No detections available yet."
            )

    except:

        st.info(
            "No detection history found."
        )

# =========================
# DETECTION PAGE
# =========================

elif selected == "Detection":

    st.title("🎥 Wildlife Detection")

    uploaded_file = st.file_uploader(
        "Upload Wildlife Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4']
    )

    if uploaded_file:

        # IMAGE DETECTION

        if uploaded_file.type.startswith('image'):

            image = Image.open(uploaded_file)

            predict = model.predict(
                image,
                conf=0.75
            )

            plotted = predict[0].plot()[:, :, ::-1]

            boxes = predict[0].boxes

            for box in boxes:

                cls = int(box.cls[0])

                confidence = float(box.conf[0])

                animal_name = model.names[cls]

                animal_type = "Non-Hazardous"

                location_name, maps_link = get_location()

                if animal_name in hazardous_animals:

                    animal_type = "Hazardous"

                    st.error(
                        f"⚠ Hazardous Animal Detected: {animal_name}"
                    )

                    st.error(
                        f"📍 Location: {location_name}"
                    )

                    st.markdown(
                        f"[🌐 Open Google Maps]({maps_link})"
                    )

                    send_sms(animal_name)

                else:

                    st.success(
                        f"Non-Hazardous Animal: {animal_name}"
                    )

                save_history(
                    animal_name,
                    animal_type,
                    round(confidence * 100, 2)
                )

            st.image(
                plotted,
                caption="Detection Result",
                use_container_width=True
            )

        # VIDEO DETECTION

        elif uploaded_file.type.startswith('video'):

            temp_file = tempfile.NamedTemporaryFile(
                delete=False
            )

            temp_file.write(
                uploaded_file.read()
            )

            temp_file.close()

            cap = cv.VideoCapture(temp_file.name)

            frame_placeholder = st.empty()

            frame_count = 0

            already_alerted = set()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                if frame_count % 2 != 0:
                    continue

                predict = model.predict(
                    frame,
                    conf=0.75
                )

                plotted = predict[0].plot()

                boxes = predict[0].boxes

                for box in boxes:

                    cls = int(box.cls[0])

                    confidence = float(box.conf[0])

                    animal_name = model.names[cls]

                    animal_type = "Non-Hazardous"

                    location_name, maps_link = get_location()

                    if animal_name in hazardous_animals:

                        animal_type = "Hazardous"

                        st.error(
                            f"⚠ Hazardous Animal Detected: {animal_name}"
                        )

                        st.error(
                            f"📍 Location: {location_name}"
                        )

                        st.markdown(
                            f"[🌐 Open Google Maps]({maps_link})"
                        )

                        if animal_name not in already_alerted:

                            send_sms(animal_name)

                            already_alerted.add(animal_name)

                    else:

                        st.success(
                            f"Non-Hazardous Animal: {animal_name}"
                        )

                    save_history(
                        animal_name,
                        animal_type,
                        round(confidence * 100, 2)
                    )

                frame_placeholder.image(
                    plotted,
                    channels="BGR",
                    caption="Live Detection"
                )

            cap.release()

            os.unlink(temp_file.name)

# =========================
# HISTORY PAGE
# =========================

elif selected == "History":

    st.title("📜 Detection History")

    df = pd.read_csv(history_file)

    st.dataframe(
        df,
        use_container_width=True
    )

# =========================
# ANALYTICS PAGE
# =========================

elif selected == "Analytics":

    st.title("📊 Detection Analytics")

    df = pd.read_csv(history_file)

    if len(df) > 0:

        chart = px.histogram(
            df,
            x="Animal",
            color="Type",
            title="Animal Detection Distribution"
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        pie = px.pie(
            df,
            names="Type",
            title="Hazardous vs Non-Hazardous"
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    else:

        st.warning(
            "No detection data available"
        )

# =========================
# ABOUT PAGE
# =========================

elif selected == "About":

    st.title("ℹ About Project")

    st.write(
        """
        This project uses YOLOv8 and Streamlit to build an
        AI-powered wildlife surveillance system capable of
        detecting hazardous and non-hazardous animals.

        Features:
        - Real-time detection
        - Hazard classification
        - SMS alerts
        - Detection history
        - Analytics dashboard
        - Google Maps integration
        - Wildlife monitoring
        """
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; padding:20px; color:gray;'>

    <h4>Project Developed By</h4>

    <b>Ankit Patil</b>,
    <b>Vaishnavi Vishwakarma</b>,
    <b>Nikhil Mulge</b>,
    <b>Anjali</b>

    <br><br>

    Department of CSE (Data Science)

    </div>
    """,
    unsafe_allow_html=True
)
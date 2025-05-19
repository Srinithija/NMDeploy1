import streamlit as st
import pyttsx3
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import random
import io
import time
import threading
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
from threading import Thread

# --- Flask Backend ---
def create_flask_app():
    flask_app = Flask(__name__)
    model = load_model("digit_recognition_model.h5")  # Make sure this file is in your deployment

    def preprocess_digits(img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((280, 280))
        img_array = np.array(img)
        gray = 255 - img_array
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return []

        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_pairs = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0])
        digits = []
        for contour, (x, y, w, h) in sorted_pairs:
            if w < 5 or h < 5:
                continue
            digit_img = thresh[y:y+h, x:x+w]
            resized = cv2.resize(digit_img, (20, 20), interpolation=cv2.INTER_AREA)
            padded = np.pad(resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
            padded = padded.astype("float32") / 255.0
            digits.append(padded.reshape(28, 28, 1))
        return digits

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        digits = preprocess_digits(file.read())
        if len(digits) == 0:
            return jsonify({"error": "No digits found"}), 200

        digits = np.array(digits)
        preds = model.predict(digits)
        predicted = np.argmax(preds, axis=1).tolist()
        confidences = (np.max(preds, axis=1) * 100).round(2).tolist()
        return jsonify({"predicted": predicted, "confidences": confidences})

    return flask_app

# Start Flask server in a separate thread
flask_app = create_flask_app()
Thread(target=lambda: flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)).start()

# --- Streamlit Frontend ---
# Set page config as the very first Streamlit command
st.set_page_config(page_title="Multi-Digit Drawing Game", page_icon="🎯", layout="centered")

# --- Speech ---
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("voice", engine.getProperty("voices")[1].id)
    engine.say(text)
    engine.runAndWait()

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# --- State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Easy"
if "target_digits" not in st.session_state:
    st.session_state.target_digits = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "high_score" not in st.session_state:
    st.session_state.high_score = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

# --- Styles ---
def set_theme():
    st.markdown(
        """ <style>
        .stApp {
            background-color: #ffe6f0;
            color: #000;
        } </style>
        """,
        unsafe_allow_html=True
    )

set_theme()

# --- Home Page ---
if st.session_state.page == "home":
    st.title("🎯 Multi-Digit Drawing Game")
    st.markdown("Choose your difficulty and try to draw the number shown!")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 Easy (1 digit)"):
            st.session_state.difficulty = "Easy"
            st.session_state.target_digits = [random.randint(0, 9)]
            st.session_state.page = "game"
            st.session_state.start_time = time.time()
            st.rerun()
    with col2:
        if st.button("🟡 Medium (3 digits)"):
            st.session_state.difficulty = "Medium"
            st.session_state.target_digits = [random.randint(0, 9) for _ in range(3)]
            st.session_state.page = "game"
            st.session_state.start_time = time.time()
            st.rerun()
    with col3:
        if st.button("🔴 Hard (5 digits)"):
            st.session_state.difficulty = "Hard"
            st.session_state.target_digits = [random.randint(0, 9) for _ in range(5)]
            st.session_state.page = "game"
            st.session_state.start_time = time.time()
            st.rerun()

    if st.button("ℹ️ Help / Instructions"):
        with st.expander("Instructions"):
            st.markdown("""
            - Select a difficulty level.
            - Draw the number shown using your mouse or touchscreen.
            - Your score and accuracy are shown.
            - Aim to improve your high score and get on the leaderboard!
            """)

# --- Game Page ---
elif st.session_state.page == "game":
    st.header("🧠 Draw the Number")

    target_number = "".join(str(d) for d in st.session_state.target_digits)
    st.subheader(f"Target: **{target_number}**")

    # Timer
    time_limit = 30
    elapsed_time = int(time.time() - st.session_state.start_time)
    remaining_time = max(0, time_limit - elapsed_time)
    st.markdown(f"⏳ Time left: **{remaining_time} seconds**")

    if remaining_time == 0:
        st.error("⏰ Time's up!")
        speak_async("Time's up!")
        if st.button("🔁 Try Again"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    # Drawing canvas
    st.markdown("✍️ Draw below:")
    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_fixed"
    )

    # Prediction Button
    if st.button("✨ Check Drawing"):
        st.session_state.attempts += 1

        if canvas.image_data is not None:
            full_img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype(np.uint8)).resize((280, 280))
            img_bytes_io = io.BytesIO()
            full_img.save(img_bytes_io, format="PNG")
            img_bytes = img_bytes_io.getvalue()
        else:
            st.warning("Draw something first.")
            st.stop()

        try:
            res = requests.post("http://localhost:5000/predict", files={"file": io.BytesIO(img_bytes)})
            result = res.json()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            speak_async("Sorry, I couldn't connect to the server.")
            st.stop()

        if "error" in result:
            st.error("❌ " + result["error"])
            speak_async("Sorry, no digits found.")
        else:
            predicted_digits = result["predicted"]
            confidences = result["confidences"]

            predicted_number = "".join(str(d) for d in predicted_digits)
            st.markdown(f"✅ You drew: **{predicted_number}**")

            # Score and feedback
            if predicted_digits == st.session_state.target_digits:
                st.success("🎉 Correct!")
                st.balloons()
                st.session_state.score += 1
                st.session_state.high_score = max(st.session_state.score, st.session_state.high_score)
                st.session_state.leaderboard.append({
                    "score": st.session_state.score,
                    "accuracy": round(100 * st.session_state.score / st.session_state.attempts, 2),
                    "time": time.strftime("%H:%M:%S")
                })
                speak_async("Great job! That's correct.")
            else:
                st.error("❌ Try again!")
                speak_async(f"You wrote {predicted_number}. But the answer is {target_number}.")

            st.markdown(f"**Score:** {st.session_state.score}")
            st.markdown(f"**High Score:** {st.session_state.high_score}")
            st.markdown(f"**Accuracy:** {100 * st.session_state.score / st.session_state.attempts:.2f}%")

            # Confidence chart
            st.markdown("### 📊 Confidence Chart")
            fig, ax = plt.subplots()
            ax.bar(range(len(predicted_digits)), confidences, tick_label=[str(d) for d in predicted_digits], color="#ff69b4")
            ax.set_xlabel("Digits")
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔁 New Set"):
            digits_count = {"Easy": 1, "Medium": 3, "Hard": 5}[st.session_state.difficulty]
            st.session_state.target_digits = [random.randint(0, 9) for _ in range(digits_count)]
            st.session_state.start_time = time.time()
            st.rerun()
    with col2:
        if st.button("🔄 Reset Score"):
            st.session_state.score = 0
            st.session_state.attempts = 0
            st.session_state.leaderboard = []
            st.rerun()
    with col3:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()

    # Leaderboard display
    if st.session_state.leaderboard:
        st.markdown("## 🏆 Leaderboard")
        st.dataframe(st.session_state.leaderboard)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🚀 Built with ❤️ by 2^ </p>", unsafe_allow_html=True)

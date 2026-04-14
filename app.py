import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from textblob import TextBlob
from pydub import AudioSegment
import speech_recognition as sr
import tempfile

# ---------------- PAGE CONFIG ----------------

st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp {
background: radial-gradient(circle at top,#1a2a6c,#b21f1f,#fdbb2d);
color:white;
}
h1,h2,h3 {color:#00fff5;}
.stButton>button {background:#00fff5;color:black;border-radius:12px;font-weight:bold;}
[data-testid="stMetric"] {
background: rgba(255,255,255,0.15);
padding:15px;
border-radius:15px;
box-shadow:0 0 10px #00fff5;
}
</style>
""",unsafe_allow_html=True)

# ---------------- SESSION ----------------

if "face" not in st.session_state: st.session_state.face="Neutral"
if "voice" not in st.session_state: st.session_state.voice="Neutral"
if "text" not in st.session_state: st.session_state.text="Neutral"

# ---------------- LOAD MODEL ----------------

model = load_model("emotion_model.h5", compile=False)
labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ---------------- FACE ----------------

def detect_face_emotion(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces)==0: return None

    faces = sorted(faces,key=lambda x:x[2]*x[3],reverse=True)
    x,y,w,h = faces[0]

    face = gray[y:y+h,x:x+w]
    face = cv2.resize(face,(64,64))
    face = face/255.0
    face = face.reshape(1,64,64,1)

    pred = model.predict(face)
    return labels[np.argmax(pred)]

# ---------------- LIVE CAM ----------------

class VideoProcessor(VideoProcessorBase):
    def recv(self,frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,3)

        if len(faces)>0:
            faces = sorted(faces,key=lambda x:x[2]*x[3],reverse=True)
            x,y,w,h = faces[0]

            face = gray[y:y+h,x:x+w]
            face = cv2.resize(face,(64,64))
            face = face/255.0
            face = face.reshape(1,64,64,1)

            pred = model.predict(face)
            emo = labels[np.argmax(pred)]

            st.session_state.face = emo

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,emo,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img,format="bgr24")

# ---------------- VOICE ----------------

def voice_emotion(audio):
    sound = AudioSegment.from_file(audio)
    temp = tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
    sound.export(temp.name,format="wav")

    r = sr.Recognizer()
    with sr.AudioFile(temp.name) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
    except:
        return "Neutral",""

    pol = TextBlob(text).sentiment.polarity
    if pol>0.4: emo="Happy"
    elif pol<-0.3: emo="Sad"
    else: emo="Neutral"

    return emo,text

# ---------------- TEXT ----------------

def text_emotion(text):
    pol = TextBlob(text).sentiment.polarity
    if pol>0.4: return "Happy"
    elif pol<-0.3: return "Sad"
    else: return "Neutral"

# ---------------- UI ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

*{font-family:Poppins;}

.stApp{
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

/* Title Glow */
h1{
text-shadow:0 0 25px cyan;
}

/* Buttons */
.stButton>button{
background:linear-gradient(45deg,#00fff5,#00c6ff);
border-radius:20px;
font-weight:bold;
padding:10px 25px;
box-shadow:0 0 15px cyan;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.1);
box-shadow:0 0 30px cyan;
}

/* Radio cards */
div[role="radiogroup"] label{
background:rgba(255,255,255,0.12);
padding:12px;
border-radius:15px;
margin:5px;
transition:0.3s;
}

div[role="radiogroup"] label:hover{
background:rgba(0,255,255,0.3);
transform:scale(1.05);
}

/* Glass Dashboard */
[data-testid="stMetric"]{
background:rgba(255,255,255,0.15);
backdrop-filter:blur(12px);
border-radius:25px;
padding:20px;
box-shadow:0 0 25px #00fff5;
transition:0.4s;
}

[data-testid="stMetric"]:hover{
transform:translateY(-8px) scale(1.05);
box-shadow:0 0 45px cyan;
}

/* File uploader */
section[data-testid="stFileUploader"]{
background:rgba(255,255,255,0.1);
padding:20px;
border-radius:20px;
box-shadow:0 0 15px cyan;
}

/* Text Area */
textarea{
border-radius:15px!important;
background:rgba(255,255,255,0.15)!important;
color:white!important;
}

/* Webcam border */
video{
border-radius:25px!important;
box-shadow:0 0 30px cyan!important;
}

/* Fade animation */
@keyframes fade{
from{opacity:0;transform:translateY(20px);}
to{opacity:1;transform:none;}
}

.stApp>*{
animation:fade 1s ease;
}

</style>
""",unsafe_allow_html=True)


st.markdown("<h1 style='text-align:center'>🧠 Mental Health Monitoring Using Emotion AI</h1>",unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Face • Voice • Text Integrated Dashboard</p>",unsafe_allow_html=True)

col1,col2 = st.columns([1,3])

with col1:
    st.markdown("### 📊 Emotion Dashboard")
    st.metric("🙂 Face",st.session_state.face)
    st.metric("🎤 Voice",st.session_state.voice)
    st.metric("✍ Text",st.session_state.text)

    moods=[st.session_state.face,st.session_state.voice,st.session_state.text]
    st.metric("🧠 Overall",max(set(moods),key=moods.count))

with col2:
    option = st.radio("Choose Module",["Face","Voice","Text"])

    if option=="Face":
        mode = st.radio("Mode",["Upload Image","Live Camera"])

        if mode=="Upload Image":
            img_file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

            if img_file:
                data = np.asarray(bytearray(img_file.read()),dtype=np.uint8)
                img = cv2.imdecode(data,1)
                st.image(img,channels="BGR",use_container_width=True)

                with st.spinner("Analyzing..."):
                    r = detect_face_emotion(img)

                if r:
                    st.session_state.face=r
                    st.success(f"Detected: {r}")
                else:
                    st.error("No Face Detected")

        else:
            webrtc_streamer(key="cam",video_processor_factory=VideoProcessor)

    if option=="Voice":
        audio = st.file_uploader("Upload Voice",type=["wav","mp3","m4a"])
        if audio:
            with st.spinner("Listening..."):
                emo,text = voice_emotion(audio)

            st.session_state.voice = emo
            st.write("📝 Text:",text)
            st.success(f"Voice Emotion: {emo}")

    if option=="Text":
        txt = st.text_area("Enter Text")
        if txt:
            emo = text_emotion(txt)
            st.session_state.text = emo
            st.success(f"Text Emotion: {emo}")
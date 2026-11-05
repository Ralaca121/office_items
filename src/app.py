import base64
import sys
from pathlib import Path

import cv2
import streamlit.components.v1 as components
import numpy as np
import streamlit as st

# Add the project root so the app can import the shared inference pipeline.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.stream_cli import load_model, infer, draw_output, DEFAULT_MODELS, DetectionSmoother

# Configure the page before creating any Streamlit UI elements.
st.set_page_config(page_title="Office Items Detector", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")


def load_css(path=Path(__file__).parent / "styles" / "main.css"):
    # Loading the app's stylesheet so the UI also works without a network connection.
    css_path = Path(path)
    if not css_path.is_file():
        st.warning(f"CSS file not found: {css_path}")
        return
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def load_start_image(path=Path(__file__).parent / "assets" / "start.png"):
    if not path.is_file():
        st.warning(f"Start image not found: {path}")
        return
    # Pass the image to CSS as a data URL rather than relying on a separate web server.
    b64 = base64.b64encode(path.read_bytes()).decode()
    st.markdown(f"<style>:root {{ --start-img: url(data:image/png;base64,{b64}); }}</style>", unsafe_allow_html=True)


load_css()
load_start_image()

# Keep the selected page and camera state across Streamlit reruns.
ss = st.session_state
ss.setdefault("page", "welcome")
ss.setdefault("streaming", False)


# Bundle the animation player and JSON files locally so menu animations work offline.
LOTTIE_JS = (Path(__file__).parent / "assets" / "lottie_light.min.js").read_text(encoding="utf-8")
ANIM_DIR = Path(__file__).parent / "styles" / "images"


def lottie(file_name, height=200):
    """Render a Lottie animation (bundled lottie-web, works offline)."""
    data = (ANIM_DIR / file_name).read_text(encoding="utf-8")
    components.html(
        f"""<style>html,body{{margin:0;background:transparent;overflow:hidden}}#a{{width:100%;height:{height}px}}</style>
        <div id="a"></div><script>{LOTTIE_JS}</script>
        <script>lottie.loadAnimation({{container:document.getElementById('a'),renderer:'svg',
        loop:true,autoplay:true,animationData:{data},
        rendererSettings:{{preserveAspectRatio:'xMidYMid meet'}}}});</script>""",
        height=height,
    )


def go(page):
    # Stop an active camera loop when leaving the live-stream page.
    ss.page = page
    ss.streaming = False


@st.cache_resource
def get_model(path):
    # Reuse loaded model weights across reruns instead of loading them for every interaction.
    return load_model(path)


def top_bar(show_cancel=False):
    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        st.markdown('<div class="home">', unsafe_allow_html=True)
        st.button("Home", on_click=go, args=("welcome",), key="home_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    if show_cancel:
        with c2:
            st.markdown('<div class="cancel">', unsafe_allow_html=True)
            st.button("Cancel", on_click=go, args=("menu",), key="cancel_btn")
            st.markdown('</div>', unsafe_allow_html=True)


def settings():
    # Shared inference settings used by the image-upload and live-stream pages.
    with st.sidebar:
        st.markdown('<h2><span class="ion ion-settings"></span>Settings</h2>', unsafe_allow_html=True)
        task = st.selectbox("Mode", ["detect"])
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)
        st.caption(f"Model: `{DEFAULT_MODELS[task]}`")
    return task, conf


# ---------- Pages ----------
def welcome():
    st.markdown("""
    <div class="hero">
      <h1>Welcome to the Object Detection App</h1>
    </div>""", unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 3, 2])
    with mid:
        st.markdown('<div class="start">', unsafe_allow_html=True)
        st.button("START", on_click=go, args=("menu",), use_container_width=True, key="start_btn")
        st.markdown('</div>', unsafe_allow_html=True)


def menu():
    top_bar()
    st.markdown("<h2 style='text-align:center'>Choose an input</h2>", unsafe_allow_html=True)
    _, a, b, _ = st.columns([1, 3, 3, 1])
    for col, key, label, page, anim in (
        (a, "up", "Upload Image", "upload", "Image Not Preview.json"),
        (b, "st", "Live Stream", "stream", "Live Icon.json"),
    ):
        with col, st.container(key=f"{key}_card"):
            st.button(label, on_click=go, args=(page,), use_container_width=True, key=f"{key}_btn")
            lottie(anim)


def upload():
    top_bar(show_cancel=True)
    task, conf = settings()
    st.subheader("🖼️ Image upload")
    file = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png"])
    if file:
        # Decode the uploaded bytes into an OpenCV BGR image for the inference pipeline.
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        model = get_model(DEFAULT_MODELS[task])
        output = infer(model, frame, task, conf)
        out = draw_output(frame.copy(), output, task, conf)
        l, r = st.columns(2)
        # OpenCV uses BGR; Streamlit expects RGB for correct display colors.
        l.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        r.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
        if task == "detect":
            kept = [d for d in output if d[1] >= conf]
            st.metric("Objects found", len(kept))
            if kept:
                st.table([{"Class": l_, "Confidence": f"{c*100:.1f}%"} for l_, c, _ in kept])
        else:
            st.metric(output[0], f"{output[1]*100:.1f}%")


def stream():
    top_bar(show_cancel=True)
    task, conf = settings()
    st.subheader("📹 Live stream")
    cam = st.number_input("Camera index", 0, 5, 0)
    st.button("▶ Start camera", on_click=lambda: ss.update(streaming=True), key="cam_start")

    if ss.streaming:
        model = get_model(DEFAULT_MODELS[task])
        cap = cv2.VideoCapture(int(cam))
        if not cap.isOpened():
            st.error("Could not open camera.")
            ss.streaming = False
            return
        holder = st.empty()
        n, last = 0, ([] if task == "detect" else ("...", 0.0))
        # Require two matching inference results to reduce one-frame false detections.
        smoother = DetectionSmoother(min_hits=2) if task == "detect" else None
        try:
            while ss.streaming:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                # Run model inference every other frame to reduce processing load.
                if n % 2 == 0:
                    last = infer(model, frame, task, conf)
                    if smoother:
                        last = smoother.update(last)
                out = draw_output(frame, last, task, conf)
                # Convert to RGB before displaying the current frame in Streamlit.
                holder.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        finally:
            cap.release()


{"welcome": welcome, "menu": menu, "upload": upload, "stream": stream}[ss.page]()
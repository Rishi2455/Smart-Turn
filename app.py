import streamlit as st
import numpy as np
import asyncio
import io
import wave
import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from model_service import SemanticAnalyzer
from asr_service import GroqASR, LocalWhisperASR

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Turn — Turn Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS for Premium Look
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
        font-weight: 300;
    }

    /* Result Cards */
    .result-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .result-card h3 {
        color: #fff;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .result-card p {
        color: #ccc;
        line-height: 1.6;
    }

    .status-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .status-turn-end {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #1a1a2e;
    }
    .status-continue {
        background: linear-gradient(135deg, #ff6d00, #ffab00);
        color: #1a1a2e;
    }
    .status-error {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: #fff;
    }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Sidebar */
    .sidebar-info {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.15);
        margin-bottom: 1rem;
    }
    .sidebar-info h4 {
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .sidebar-info p {
        color: #aaa;
        font-size: 0.85rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Cache Models (load once with Streamlit caching)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Turn Detection Model...")
def load_semantic_analyzer():
    return SemanticAnalyzer()


@st.cache_resource(show_spinner="Loading ASR Service...")
def load_asr_service():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return GroqASR(api_key=groq_key)
    else:
        return LocalWhisperASR(model_size="base")


@st.cache_resource(show_spinner="Loading Voice Activity Detection...")
def load_vad_model():
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True,
    )
    vad_model.eval()
    return vad_model


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────
def convert_audio_to_float32(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert uploaded audio (WAV) bytes to float32 numpy array at 16kHz mono."""
    try:
        wav_io = io.BytesIO(audio_bytes)
        with wave.open(wav_io, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw_data = wav_file.readframes(n_frames)

        if sample_width == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert stereo to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample if needed (simple decimation — good enough for demo)
        if framerate != sample_rate:
            ratio = sample_rate / framerate
            target_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
            audio = audio[indices]

        return audio
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return np.array([], dtype=np.float32)


def run_vad(vad_model, audio_float32: np.ndarray, sample_rate: int = 16000) -> dict:
    """Run VAD on audio and return speech statistics."""
    chunk_size = 512
    num_chunks = len(audio_float32) // chunk_size

    speech_chunks = 0
    total_chunks = max(num_chunks, 1)
    max_prob = 0.0

    for i in range(num_chunks):
        chunk = audio_float32[i * chunk_size : (i + 1) * chunk_size]
        tensor = torch.from_numpy(chunk).unsqueeze(0)
        with torch.no_grad():
            prob = vad_model(tensor, sample_rate).item()
        if prob > max_prob:
            max_prob = prob
        if prob > 0.3:
            speech_chunks += 1

    speech_ratio = speech_chunks / total_chunks if total_chunks > 0 else 0
    rms = float(np.sqrt(np.mean(audio_float32**2))) if len(audio_float32) > 0 else 0

    return {
        "speech_ratio": speech_ratio,
        "max_prob": max_prob,
        "rms": rms,
        "has_speech": speech_chunks > 0,
        "speech_chunks": speech_chunks,
        "total_chunks": total_chunks,
    }


async def transcribe_audio(asr_service, audio_float32: np.ndarray) -> str:
    """Transcribe audio using the ASR service."""
    if len(audio_float32) < 3200:
        return ""
    text = await asr_service.transcribe(audio_float32)
    return text.strip() if text else ""


async def check_turn(analyzer: SemanticAnalyzer, text: str) -> dict:
    """Check turn completion using the semantic analyzer."""
    is_complete, confidence = await analyzer.is_sentence_complete(text)
    status = "turn_end" if is_complete else "continue"
    reason = "semantic_complete" if is_complete else "semantic_incomplete"
    return {
        "is_complete": is_complete,
        "confidence": confidence,
        "status": status,
        "reason": reason,
    }


def render_result_card(status: str, confidence: float, transcript: str, reason: str, vad_info: dict = None):
    """Render a beautiful result card."""
    badge_class = "status-turn-end" if status == "turn_end" else "status-continue"
    badge_label = "🛑 TURN END" if status == "turn_end" else "🔄 CONTINUE"
    conf_pct = confidence * 100
    conf_color = "#00e676" if status == "turn_end" else "#ffab00"

    html = f"""
    <div class="result-card">
        <h3>🎯 Analysis Result</h3>
        <p><strong>Transcript:</strong> <em>"{transcript}"</em></p>
        <p>
            <strong>Decision:</strong>
            <span class="status-badge {badge_class}">{badge_label}</span>
        </p>
        <p><strong>Confidence:</strong> {conf_pct:.1f}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {conf_pct}%; background: {conf_color};"></div>
        </div>
        <p style="margin-top: 0.8rem; font-size: 0.85rem; color: #999;">
            <strong>Reason:</strong> {reason}
        </p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Main Layout
# ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🎙️ Smart Turn</h1>
    <p>Intelligent Turn-End Detection for Conversational AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h4>ℹ️ About</h4>
        <p>
            Smart Turn uses a fine-tuned NLP model to detect whether a user has finished
            speaking (turn-end) or is still mid-sentence (continue).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    asr_source = st.selectbox(
        "ASR Engine",
        ["Auto (Groq if key available, else Local Whisper)"],
        index=0,
    )
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + Transformers")

# Load models
semantic_analyzer = load_semantic_analyzer()
asr_service = load_asr_service()
vad_model = load_vad_model()

# ──────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────
tab_audio, tab_text = st.tabs(["🎤 Audio Analysis", "📝 Text Analysis"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Audio Tab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_audio:
    st.markdown("#### Record or upload audio to analyze turn completion")

    col_record, col_upload = st.columns(2)

    with col_record:
        st.markdown("##### 🎙️ Record Audio")
        audio_value = st.audio_input("Record a voice clip", label_visibility="collapsed")

    with col_upload:
        st.markdown("##### 📁 Upload Audio")
        uploaded_file = st.file_uploader(
            "Upload a WAV file",
            type=["wav"],
            label_visibility="collapsed",
        )

    # Determine audio source
    audio_bytes = None
    source_label = ""

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        source_label = "Recorded Audio"
    elif uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        source_label = "Uploaded File"

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("🔍 Analyze Audio", type="primary", use_container_width=True, key="analyze_audio"):
            with st.spinner("Processing audio..."):
                # Step 1: Convert to float32
                audio_float32 = convert_audio_to_float32(audio_bytes)

                if len(audio_float32) == 0:
                    st.error("Could not process audio. Please try a different file.")
                else:
                    # Step 2: VAD
                    vad_info = run_vad(vad_model, audio_float32)

                    if not vad_info["has_speech"]:
                        st.warning("⚠️ No speech detected in the audio. Try speaking louder or closer to the mic.")
                    else:
                        # Step 3: Transcribe
                        with st.spinner("Transcribing with ASR..."):
                            transcript = asyncio.run(transcribe_audio(asr_service, audio_float32))

                        if not transcript:
                            st.warning("⚠️ Could not transcribe audio. The clip might be too short or noisy.")
                        else:
                            # Step 4: Turn Detection
                            with st.spinner("Analyzing turn completion..."):
                                result = asyncio.run(check_turn(semantic_analyzer, transcript))

                            render_result_card(
                                status=result["status"],
                                confidence=result["confidence"],
                                transcript=transcript,
                                reason=result["reason"],
                                vad_info=vad_info,
                            )
    else:
        st.info("👆 Record or upload audio above to get started.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text Tab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_text:
    st.markdown("#### Type text to check if the sentence is complete")

    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="e.g. 'I want to book a flight to' or 'My number is 9876543210'",
        height=120,
    )

    if st.button("🔍 Check Turn", type="primary", use_container_width=True, key="check_text"):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                result = asyncio.run(check_turn(semantic_analyzer, text_input.strip()))

            render_result_card(
                status=result["status"],
                confidence=result["confidence"],
                transcript=text_input.strip(),
                reason=result["reason"],
            )

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666; font-size:0.8rem;'>"
    "Smart Turn v1.0 — Powered by Transformers, Silero VAD & Groq ASR"
    "</p>",
    unsafe_allow_html=True,
)

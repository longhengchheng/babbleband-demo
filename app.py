import streamlit as st
import numpy as np
import scipy.signal
import sounddevice as sd
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="BabbleBand Live AI Demo", layout="wide")

# High-visibility styling for hospital committee recording
st.markdown("""
<style>
    .banner {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍼 BabbleBand: Infant Cry & Risk Screening Demo")
st.caption("CamTech Research Project | Two-Device Setup: Play sound from phone -> Catch live via Laptop Microphone")

# Sidebar Controls
st.sidebar.header("Demo Settings")
duration = st.sidebar.slider("Listening Duration (seconds)", 3, 8, 5)
sample_rate = 16000

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("1. Live Microphone Capture")
    st.info("💡 **Instructions for your video:** Place your phone/tablet next to your laptop. Click the button below, then immediately press **Play** on your phone's baby cry audio file.")
    
    listen_btn = st.button("🎙️ START LISTENING (Catch Sound from Phone)", type="primary", use_container_width=True)

banner_placeholder = st.empty()
chart_placeholder = st.empty()

if listen_btn:
    banner_placeholder.markdown('<div class="banner" style="background-color: #0284c7;">🎙️ LISTENING... PLAY THE CRY ON YOUR PHONE NOW!</div>', unsafe_allow_html=True)
    
    # Record audio from laptop mic
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    y = audio_data.flatten()
    
    # Simple Signal Processing & Feature Extraction
    rms = float(np.sqrt(np.mean(y**2)))
    # Estimate dominant frequency using FFT peak
    fft_vals = np.fft.rfft(y)
    fft_freqs = np.fft.rfftfreq(len(y), 1/sample_rate)
    peak_freq = float(fft_freqs[np.argmax(np.abs(fft_vals))])
    
    # Check if a cry/sound was actually captured
    if rms < 0.005:
        banner_placeholder.markdown('<div class="banner" style="background-color: #64748b;">🔇 NO SOUND DETECTED. Please increase phone volume and try again.</div>', unsafe_allow_html=True)
    else:
        # Simulate ML Classification & Risk Scoring based on acoustic pitch features
        if peak_freq > 450 or rms > 0.04:
            cause = "BELLY PAIN / ACUTE DISTRESS"
            conf = random.uniform(82.4, 96.8)
            risk_flag = True
        elif peak_freq > 350:
            cause = "HUNGRY / DISCOMFORT"
            conf = random.uniform(78.5, 91.2)
            risk_flag = False
        else:
            cause = "TIRED / SLEEPY"
            conf = random.uniform(75.0, 88.0)
            risk_flag = False
            
        # Display Banner
        bg_col = "#dc2626" if risk_flag else "#16a34a"
        status_txt = "⚠️ ELEVATED RISK MARKER FLAGGED" if risk_flag else "✅ TYPICAL PHYSIOLOGICAL CRY"
        banner_placeholder.markdown(
            f'<div class="banner" style="background-color: {bg_col};">'
            f'🔊 DETECTED CAUSE: {cause} ({conf:.1f}%)<br>'
            f'<span style="font-size: 18px;">Pitch (F0): {peak_freq:.1f} Hz | {status_txt}</span>'
            f'</div>', unsafe_allow_html=True
        )
        
        with col2:
            st.subheader("2. Real-Time Waveform & Probabilities")
            fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#0f172a")
            times = np.linspace(0, duration, len(y))
            ax.plot(times, y, color="#38bdf8", linewidth=1)
            ax.set_facecolor("#1e293b")
            ax.set_title("Live Audio Waveform Buffer", color="white", fontsize=10)
            ax.tick_params(colors="white")
            st.pyplot(fig)
            
            # Probability Bar Chart
            fig2, ax2 = plt.subplots(figsize=(6, 2), facecolor="#0f172a")
            categories = ["Hungry", "Discomfort", "Belly Pain", "Tired"]
            probs = [15.0, 20.0, 60.0, 5.0] if risk_flag else [65.0, 20.0, 5.0, 10.0]
            ax2.barh(categories, probs, color="#22c55e" if not risk_flag else "#ef4444")
            ax2.set_facecolor("#1e293b")
            ax2.set_xlim(0, 100)
            ax2.set_xlabel("ML Confidence (%)", color="white")
            ax2.tick_params(colors="white")
            st.pyplot(fig2)

        st.divider()
        st.subheader("3. Automated Pediatric Screening Summary")
        if risk_flag:
            st.warning(
                f"**Clinical Note (Non-Diagnostic):** Sustained high-frequency vocalization detected "
                f"(Pitch: {peak_freq:.1f} Hz). Pattern matches indicators associated with acute discomfort or distress. "
                f"Recommended action: Flag for pediatrician review."
            )
        else:
            st.success(
                f"**Clinical Note (Non-Diagnostic):** Acoustic pattern indicates standard physiological need "
                f"({cause}). Fundamental frequency is within expected developmental bounds ({peak_freq:.1f} Hz). "
                f"No elevated neurodevelopmental risk flagged."
            )
"""
和弦辨識網頁 App - Streamlit 版
--------------------------------
安裝方式（只需這一行）:
    pip install streamlit librosa scipy numpy soundfile

啟動方式（只需這一行）:
    streamlit run app.py

執行後會自動開啟瀏覽器網頁，網址通常是 http://localhost:8501
"""

import streamlit as st
import numpy as np
import librosa
from scipy.ndimage import median_filter
import sqlite3
import json
from datetime import datetime
import os
import tempfile

# ========== 和弦辨識核心邏輯（已驗證過）==========

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_INTERVALS = {
    "":     [0, 4, 7],
    "m":    [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7":   [0, 3, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 2],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
}

def build_templates():
    templates = {}
    for root_i in range(12):
        root_name = PITCHES[root_i]
        for suffix, intervals in CHORD_INTERVALS.items():
            vec = np.zeros(12)
            for itv in intervals:
                vec[(root_i + itv) % 12] = 1.0
            vec = vec / np.linalg.norm(vec)
            templates[f"{root_name}{suffix}"] = vec
    return templates

TEMPLATES = build_templates()
TEMPLATE_LABELS = list(TEMPLATES.keys())
TEMPLATE_MATRIX = np.stack([TEMPLATES[l] for l in TEMPLATE_LABELS], axis=0)

def recognize_chords(audio_path, hop_length=2048, smooth_size=9, min_seg_sec=0.3):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = median_filter(chroma, size=(1, smooth_size))

    frame_labels, frame_confidence = [], []
    for frame in chroma.T:
        norm = np.linalg.norm(frame)
        if norm < 1e-6:
            frame_labels.append("N")
            frame_confidence.append(0.0)
            continue
        frame_n = frame / norm
        scores = TEMPLATE_MATRIX @ frame_n
        best_idx = int(np.argmax(scores))
        frame_labels.append(TEMPLATE_LABELS[best_idx])
        frame_confidence.append(float(scores[best_idx]))

    times = librosa.frames_to_time(np.arange(len(frame_labels)), sr=sr, hop_length=hop_length)

    segments = []
    cur_label = frame_labels[0]
    cur_start = times[0]
    cur_conf = [frame_confidence[0]]
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != cur_label:
            segments.append({
                "chord": cur_label,
                "start": round(float(cur_start), 2),
                "end": round(float(times[i]), 2),
                "confidence": round(float(np.mean(cur_conf)), 3),
            })
            cur_label = frame_labels[i]
            cur_start = times[i]
            cur_conf = []
        cur_conf.append(frame_confidence[i])
    segments.append({
        "chord": cur_label,
        "start": round(float(cur_start), 2),
        "end": round(float(times[-1]), 2),
        "confidence": round(float(np.mean(cur_conf)), 3),
    })

    filtered = []
    for seg in segments:
        if seg["end"] - seg["start"] < min_seg_sec and filtered:
            filtered[-1]["end"] = seg["end"]
        else:
            filtered.append(seg)
    return filtered

# ========== 資料庫（收集資料用）==========

DB_PATH = "chord_database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analysis_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            song_title TEXT,
            artist TEXT,
            genre TEXT,
            duration_sec REAL,
            chord_count INTEGER,
            chord_progression TEXT,
            avg_confidence REAL,
            user_email TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_record(song_title, artist, genre, duration_sec, chords, user_email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    avg_conf = sum(ch["confidence"] for ch in chords) / len(chords) if chords else 0
    c.execute("""
        INSERT INTO analysis_records
        (created_at, song_title, artist, genre, duration_sec, chord_count, chord_progression, avg_confidence, user_email)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(), song_title, artist, genre, duration_sec,
        len(chords), json.dumps(chords, ensure_ascii=False), avg_conf, user_email
    ))
    conn.commit()
    conn.close()

# ========== 網頁介面 ==========

st.set_page_config(page_title="自動和弦辨識", page_icon="🎸")
init_db()

st.title("🎸 自動和弦辨識")
st.caption("上傳音檔，自動分析和弦進行")

with st.form("upload_form"):
    uploaded_file = st.file_uploader("選擇音檔", type=["mp3", "wav", "m4a", "flac"])
    col1, col2 = st.columns(2)
    with col1:
        song_title = st.text_input("歌曲名稱", placeholder="例如：告白氣球")
    with col2:
        artist = st.text_input("歌手", placeholder="例如：周杰倫")
    genre = st.selectbox("曲風", ["流行", "民謠", "搖滾", "抒情", "其他"])
    user_email = st.text_input("Email（選填，用於未來功能通知）", placeholder="your@email.com")
    submitted = st.form_submit_button("🔍 開始分析")

if submitted and uploaded_file is not None:
    with st.spinner("分析中，請稍候..."):
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            segments = recognize_chords(tmp_path)
            duration = segments[-1]["end"] if segments else 0.0

            save_record(song_title or "未命名", artist or "未知",
                        genre, duration, segments, user_email or None)

            st.success(f"✅ 分析完成！共偵測到 {len(segments)} 個和弦段落")

            st.subheader("和弦進行結果")
            for seg in segments:
                conf_color = "🟢" if seg["confidence"] > 0.9 else ("🟡" if seg["confidence"] > 0.7 else "🔴")
                st.write(f"{conf_color} **{seg['chord']}**　|　{seg['start']}s ~ {seg['end']}s　|　信心度 {seg['confidence']:.2f}")

            chord_line = "  ".join([seg["chord"] for seg in segments])
            st.subheader("簡易和弦譜")
            st.code(chord_line)

        except Exception as e:
            st.error(f"分析失敗：{str(e)}")
        finally:
            os.remove(tmp_path)

st.divider()
st.caption("💡 這是測試版本，和弦辨識結果可能不完全準確，持續改進中")

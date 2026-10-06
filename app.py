"""
和弦辨識網頁 App - Streamlit 版（91吉他譜樣式 + 段落標示）
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
import html
import re

# ========== 和弦辨識核心邏輯 ==========

# ========== 和弦辨識核心邏輯 ==========

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 加上 penalty（複雜度懲罰），數字越大代表越容易被懲罰、越不容易被誤選
CHORD_TYPES = {
    "":     {"intervals": [0, 4, 7],      "penalty": 0.00},
    "m":    {"intervals": [0, 3, 7],      "penalty": 0.00},
    "7":    {"intervals": [0, 4, 7, 10],  "penalty": 0.03},
    "m7":   {"intervals": [0, 3, 7, 10],  "penalty": 0.03},
    "maj7": {"intervals": [0, 4, 7, 11],  "penalty": 0.05},
    "sus2": {"intervals": [0, 2, 7],      "penalty": 0.05},
    "sus4": {"intervals": [0, 5, 7],      "penalty": 0.05},
    "add9": {"intervals": [0, 4, 7, 2],   "penalty": 0.06},
    "dim":  {"intervals": [0, 3, 6],      "penalty": 0.06},
    "aug":  {"intervals": [0, 4, 8],      "penalty": 0.06},
}

def build_templates():
    templates = {}
    for root_i in range(12):
        root_name = PITCHES[root_i]
        for suffix, cfg in CHORD_TYPES.items():
            vec = np.zeros(12)
            for itv in cfg["intervals"]:
                vec[(root_i + itv) % 12] = 1.0
            vec = vec / np.linalg.norm(vec)
            templates[f"{root_name}{suffix}"] = {
                "vector": vec,
                "penalty": cfg["penalty"],
            }
    return templates

TEMPLATES = build_templates()
TEMPLATE_LABELS = list(TEMPLATES.keys())
TEMPLATE_MATRIX = np.stack([TEMPLATES[l]["vector"] for l in TEMPLATE_LABELS], axis=0)
TEMPLATE_PENALTIES = np.array([TEMPLATES[l]["penalty"] for l in TEMPLATE_LABELS])

# ---------- 調性偵測（新增） ----------

MAJOR_SCALE_DEGREES = [0, 2, 4, 5, 7, 9, 11]
DEGREE_CHORD_TYPES = ["", "m", "m", "", "", "m", "dim"]

def get_diatonic_chords(key_root_i):
    diatonic = []
    for deg, ctype in zip(MAJOR_SCALE_DEGREES, DEGREE_CHORD_TYPES):
        root_i = (key_root_i + deg) % 12
        diatonic.append(PITCHES[root_i] + ctype)
    return diatonic

def detect_key(chroma):
    avg_chroma = np.mean(chroma, axis=1)
    avg_chroma = avg_chroma / (np.linalg.norm(avg_chroma) + 1e-9)
    best_key, best_score = 0, -1
    for key_i in range(12):
        mask = np.zeros(12)
        for deg in MAJOR_SCALE_DEGREES:
            mask[(key_i + deg) % 12] = 1
        score = np.dot(avg_chroma, mask)
        if score > best_score:
            best_score, best_key = score, key_i
    return best_key

# ---------- 時間平滑（新增） ----------

def smooth_labels(labels, window=5):
    def complexity_rank(chord):
        for suffix in ["add9", "aug", "dim", "maj7", "sus4", "sus2", "m7", "7", "m", ""]:
            if chord.endswith(suffix) and chord != "N":
                return len(suffix)
        return 99 if chord != "N" else -1

    smoothed = list(labels)
    half = window // 2
    for i in range(len(labels)):
        start = max(0, i - half)
        end = min(len(labels), i + half + 1)
        window_labels = labels[start:end]
        counts = {}
        for l in window_labels:
            counts[l] = counts.get(l, 0) + 1
        max_count = max(counts.values())
        candidates = [c for c, cnt in counts.items() if cnt == max_count]
        smoothed[i] = min(candidates, key=complexity_rank)
    return smoothed

# ---------- 主辨識函式（修改） ----------

def recognize_chords(audio_path, hop_length=2048, smooth_size=9, min_seg_sec=0.5,
                      diatonic_bonus=0.08, smooth_window=5):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = median_filter(chroma, size=(1, smooth_size))

    # 新增：先偵測整首歌的調性
    key_root_i = detect_key(chroma)
    diatonic_chords = set(get_diatonic_chords(key_root_i))
    diatonic_bonus_vec = np.array([
        diatonic_bonus if label in diatonic_chords else 0.0
        for label in TEMPLATE_LABELS
    ])

    frame_labels, frame_confidence = [], []
    for frame in chroma.T:
        norm = np.linalg.norm(frame)
        if norm < 1e-6:
            frame_labels.append("N")
            frame_confidence.append(0.0)
            continue
        frame_n = frame / norm
        # 新增：扣掉複雜度懲罰、加上調性加分
        scores = TEMPLATE_MATRIX @ frame_n - TEMPLATE_PENALTIES + diatonic_bonus_vec
        best_idx = int(np.argmax(scores))
        frame_labels.append(TEMPLATE_LABELS[best_idx])
        frame_confidence.append(float(scores[best_idx]))

    # 新增：時間平滑，過濾瞬間誤判
    frame_labels = smooth_labels(frame_labels, window=smooth_window)

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

# ========== 歌詞解析：區分標籤行 / 歌詞行 ==========

def parse_lyrics_with_tags(lyrics_text):
    """
    [前奏]、[副歌] 這種整行只有標籤的 → type: tag
    (男)歌詞內容 這種行首有角色標籤的 → type: lyric, 附帶 prefix_tag
    空白行 → type: blank
    """
    lines = lyrics_text.split("\n")
    parsed = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            parsed.append({"type": "blank", "content": ""})
            continue

        tag_only_match = re.fullmatch(r"[\[\(][^\]\)]*[\]\)]", stripped)
        if tag_only_match:
            parsed.append({"type": "tag", "content": stripped})
            continue

        prefix_match = re.match(r"^([\[\(][^\]\)]*[\]\)])(.*)", stripped)
        if prefix_match:
            tag_part = prefix_match.group(1)
            lyric_part = prefix_match.group(2).strip()
            parsed.append({"type": "lyric", "content": lyric_part, "prefix_tag": tag_part})
            continue

        parsed.append({"type": "lyric", "content": stripped, "prefix_tag": None})

    return parsed

# ========== 和弦對齊 + 91譜樣式渲染 ==========

def align_chords_to_lyrics(chords, parsed_lines, total_duration):
    flat_chars = []
    for li, item in enumerate(parsed_lines):
        if item["type"] != "lyric":
            continue
        for ci, ch in enumerate(item["content"]):
            if ch != " ":
                flat_chars.append((li, ci, ch))

    total_chars = len(flat_chars)
    if total_chars == 0:
        return {}

    chord_anchor_points = []
    for ch in chords:
        if ch["chord"] == "N":
            continue
        ratio = ch["start"] / total_duration if total_duration > 0 else 0
        char_index = min(int(ratio * total_chars), total_chars - 1)
        chord_anchor_points.append((char_index, ch["chord"]))

    marks_by_line = {
        li: [None] * len(item["content"])
        for li, item in enumerate(parsed_lines) if item["type"] == "lyric"
    }

    for char_index, chord_name in chord_anchor_points:
        li, ci, _ = flat_chars[char_index]
        if marks_by_line[li][ci] is None:
            marks_by_line[li][ci] = chord_name

    return marks_by_line


def render_91_style_html(parsed_lines, marks_by_line):
    blocks = []
    for li, item in enumerate(parsed_lines):
        if item["type"] == "blank":
            blocks.append("<div style='height:14px;'></div>")
            continue

        if item["type"] == "tag":
            blocks.append(
                f"<div style='color:#888; font-size:13px; margin:6px 0 2px 0;'>{html.escape(item['content'])}</div>"
            )
            continue

        prefix = item.get("prefix_tag")
        content = item["content"]
        marks = marks_by_line.get(li, [None] * len(content))

        row_html = ""
        if prefix:
            row_html += f"<span style='color:#888; font-size:13px; margin-right:4px;'>{html.escape(prefix)}</span>"

        chord_cells = ""
        lyric_cells = ""
        for ch, mark in zip(content, marks):
            if ch == " ":
                chord_cells += "<td style='width:8px;'></td>"
                lyric_cells += "<td style='width:8px;'></td>"
                continue
            chord_text = mark if mark else ""
            chord_cells += f"<td style='padding:0 2px; color:#1a73e8; font-weight:bold; font-size:13px; text-align:center; white-space:nowrap;'>{html.escape(chord_text)}</td>"
            lyric_cells += f"<td style='padding:0 2px; text-align:center; font-size:16px; color:#1a1a1a;'>{html.escape(ch)}</td>"

        table = (
            "<table style='border-collapse:collapse; display:inline-table; vertical-align:middle;'>"
            f"<tr>{chord_cells}</tr><tr>{lyric_cells}</tr>"
            "</table>"
        )
        blocks.append(f"<div style='margin-bottom:2px;'>{row_html}{table}</div>")

    return "".join(blocks)


# ========== 資料庫 ==========

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

    # 自動偵測並補上後來新增的欄位，避免舊資料庫缺欄位報錯
    c.execute("PRAGMA table_info(analysis_records)")
    existing_columns = [row[1] for row in c.fetchall()]

    required_columns = {
        "has_lyrics": "BOOLEAN DEFAULT 0"
    }

    for col_name, col_def in required_columns.items():
        if col_name not in existing_columns:
            c.execute(f"ALTER TABLE analysis_records ADD COLUMN {col_name} {col_def}")

    conn.commit()
    conn.close()


def save_record(song_title, artist, genre, duration_sec, chords, user_email, has_lyrics):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    avg_conf = sum(ch["confidence"] for ch in chords) / len(chords) if chords else 0
    c.execute("""
        INSERT INTO analysis_records
        (created_at, song_title, artist, genre, duration_sec, chord_count, chord_progression, avg_confidence, user_email, has_lyrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(), song_title, artist, genre, duration_sec,
        len(chords), json.dumps(chords, ensure_ascii=False), avg_conf, user_email, has_lyrics
    ))
    conn.commit()
    conn.close()

# ========== 網頁介面 ==========

st.set_page_config(page_title="自動和弦辨識", page_icon="🎸", layout="wide")
init_db()

st.title("🎸 自動和弦辨識 - 91譜樣式")
st.caption("上傳音檔 + 貼上歌詞，自動產生和弦對照歌詞譜（支援段落標示）")

with st.expander("📖 歌詞格式說明（點我展開）"):
    st.markdown("""
    支援以下標記方式，讓排版更接近91吉他譜：

    - `[前奏]`、`[副歌]`、`[間奏]` → 整行只放這種標籤，會顯示為獨立的段落提示
    - `(男)`、`(女)`、`(合)` → 放在歌詞行最前面，會顯示為該行的角色標示
    - 空白行 → 會變成段落間的留白

    範例：
    ```
    [前奏]
    (男)你還愛我嗎 你還愛我嗎 你怪我合不爭氣想回到你身旁

    [副歌]
    (合)也許這一切都是最好的安排 但我無法看著你難開
    ```
    """)

with st.form("upload_form"):
    uploaded_file = st.file_uploader("選擇音檔", type=["mp3", "wav", "m4a", "flac"])
    col1, col2 = st.columns(2)
    with col1:
        song_title = st.text_input("歌曲名稱", placeholder="例如：往加納共和國離婚")
    with col2:
        artist = st.text_input("歌手", placeholder="例如：菲道爾")
    genre = st.selectbox("曲風", ["流行", "民謠", "搖滾", "抒情", "其他"])
    lyrics_text = st.text_area(
        "貼上歌詞（可用 [標籤] 和 (角色) 格式，詳見上方說明）",
        height=180,
        placeholder="[前奏]\n(男)你還愛我嗎 你還愛我嗎 你怪我合不爭氣想回到你身旁\n\n[副歌]\n(合)也許這一切都是最好的安排"
    )
    user_email = st.text_input("Email（選填）", placeholder="your@email.com")
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

            parsed_lines = parse_lyrics_with_tags(lyrics_text) if lyrics_text.strip() else []
            has_lyrics = len(parsed_lines) > 0

            save_record(song_title or "未命名", artist or "未知", genre,
                        duration, segments, user_email or None, has_lyrics)

            st.success(f"✅ 分析完成！共偵測到 {len(segments)} 個和弦段落")

            if parsed_lines:
                st.subheader("📜 和弦歌詞對照譜")
                marks_by_line = align_chords_to_lyrics(segments, parsed_lines, duration)
                rendered_html = render_91_style_html(parsed_lines, marks_by_line)
                st.markdown(
                    f"<div style='line-height:1.8; padding:16px; background:#fafafa; border-radius:8px;'>{rendered_html}</div>",
                    unsafe_allow_html=True
                )
                st.caption("⚠️ 和弦位置是依時間比例自動對齊，可能與實際彈奏點略有誤差")
            else:
                st.info("💡 沒有貼歌詞時，只顯示和弦時間軸列表（貼上歌詞可以產生91譜樣式對照）")

            with st.expander("查看詳細和弦時間軸"):
                for seg in segments:
                    conf_color = "🟢" if seg["confidence"] > 0.9 else ("🟡" if seg["confidence"] > 0.7 else "🔴")
                    st.write(f"{conf_color} **{seg['chord']}** | {seg['start']}s ~ {seg['end']}s | 信心度 {seg['confidence']:.2f}")

            chord_line = "  ".join([seg["chord"] for seg in segments if seg["chord"] != "N"])
            st.subheader("簡易和弦序列")
            st.code(chord_line)

        except Exception as e:
            st.error(f"分析失敗：{str(e)}")
        finally:
            os.remove(tmp_path)

st.divider()
st.caption("💡 這是測試版本，和弦辨識結果可能不完全準確，持續改進中")

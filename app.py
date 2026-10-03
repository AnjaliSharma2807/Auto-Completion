import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Auto-Completion",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

html, body, [class*="css"]{
    font-family: "Segoe UI", sans-serif;
}

/* ==========================
   APP BACKGROUND
========================== */

.stApp{
    background:linear-gradient(
        135deg,
        #F8FAFC,
        #F1F5F9
    );
}

/* ==========================
   SIDEBAR
========================== */

section[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #081225,
        #102A43,
        #1B365D
    );
    border-right:1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label{
    color:white !important;
}

/* Prediction History Cards */

.sidebar-card{
    background:linear-gradient(
        135deg,
        #FFFFFF,
        #F0FDFA
    );

    border:1px solid #99F6E4;

    border-radius:16px;

    padding:14px;

    margin-bottom:12px;

    color:#0F766E !important;

    font-weight:700;

    font-size:15px;

    box-shadow:
        0 5px 15px rgba(20,184,166,0.12);

    transition:all .3s ease;
}

.sidebar-card *{
    color:#0F766E !important;
}

.sidebar-card:hover{
    transform:translateY(-3px);
    box-shadow:
        0 10px 25px rgba(20,184,166,0.18);
}

/* ==========================
   MAIN CONTAINER
========================== */

.main-container{
    background:white;

    border-radius:28px;

    padding:30px;

    border:1px solid #E2E8F0;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.05),
        0 20px 50px rgba(20,184,166,0.08);

    margin-bottom:20px;
}

/* ==========================
   HEADER
========================== */

.title-row{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:15px;
}

.main-title{
    font-size:54px;
    font-weight:800;
    color:#0F172A;
    letter-spacing:-1px;
}

.model-badge{
    background:#E6FFFA;

    color:#0F766E;

    padding:12px 20px;

    border-radius:14px;

    border:1px solid #99F6E4;

    font-size:18px;

    font-weight:700;
}

/* ==========================
   KPI CARDS
========================== */

.metric-card{
    background:rgba(20, 184, 166, 0.10);

    border:2px solid #14B8A6;

    border-radius:18px;

    padding:22px 15px;

    text-align:center;

    box-shadow:
        0 8px 25px rgba(0,0,0,.05);

    transition:all .3s ease;
}

.metric-card:hover{
    transform:translateY(-4px);

    box-shadow:
        0 12px 30px rgba(20,184,166,0.12);
}

.metric-title{
    color:#0F766E;

    font-size:14px;

    font-weight:600;

    margin-bottom:10px;

    text-align:center;
}

.metric-value{
    color:#0F172A;

    font-size:30px;

    font-weight:800;

    text-align:center;

    margin-top:5px;

    line-height:1.2;
}


/* ==========================
   INPUT LABEL
========================== */

.editor-label{
    color:#334155;

    font-size:24px;

    font-weight:700;

    margin-bottom:10px;
}

/* ==========================
   TEXT AREA
========================== */

.stTextArea textarea{

    background:#FFFFFF !important;

    color:#000000 !important;

    caret-color:#14B8A6 !important;

    border:2px solid #DCE7EC !important;

    border-radius:20px !important;

    padding:20px !important;

    font-size:20px !important;

    font-weight:500 !important;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.05) !important;

    min-height:300px !important;
}

.stTextArea textarea:focus{

    border:2px solid #14B8A6 !important;

    box-shadow:
        0 0 0 4px rgba(20,184,166,0.15) !important;
}

.stTextArea textarea::placeholder{
    color:#94A3B8 !important;
    opacity:1 !important;
}

/* ==========================
   BUTTON
========================== */

.stButton button{

    width:100%;

    height:60px;

    border:none;

    border-radius:16px;

    background:linear-gradient(
        135deg,
        #14B8A6,
        #0F766E
    );

    color:white;

    font-size:20px;

    font-weight:700;

    transition:.3s;
}

.stButton button:hover{

    transform:translateY(-2px);

    box-shadow:
        0 10px 20px rgba(20,184,166,0.25);
}

/* ==========================
   PREDICTION CARD
========================== */

.suggestion-box{

    background:linear-gradient(
        135deg,
        #ECFDF5,
        #F0FDFA
    );

    border:2px solid #A7F3D0;

    border-radius:20px;

    padding:24px;

    margin-top:20px;

    color:#065F46;

    font-size:24px;

    font-weight:700;

    box-shadow:
        0 10px 25px rgba(16,185,129,.15);
}

.confidence-badge{

    background:#0F766E;

    color:white;

    padding:8px 14px;

    border-radius:12px;

    font-size:14px;

    font-weight:700;
}

/* ==========================
   STATUS BAR
========================== */

.status-bar{

    background:#F8FAFC;

    border:1px solid #E2E8F0;

    border-radius:16px;

    padding:16px;

    margin-top:20px;

    color:#334155;

    font-weight:600;

    box-shadow:
        0 4px 15px rgba(0,0,0,0.04);
}

/* ==========================
   PROGRESS BAR
========================== */

.stProgress > div > div > div > div{
    background:#14B8A6 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "lstm_model (1).h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN_PATH = "max_len (1).pkl"

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(MAXLEN_PATH, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🤖Auto-Completion")

    st.markdown("### Recent Predictions")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet")

    for item in reversed(st.session_state.history[-8:]):
        st.markdown(
            f"""
            <div class="sidebar-card">
            {item}
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# HEADER
# =====================================================

st.markdown(
f"""
<div class="main-container">

<div class="title-row">

<div class="main-title">
🤖 AI-Powered Auto-Text Generator 
</div>

<div class="model-badge">
⚙️ Model : LSTM-{len(tokenizer.word_index)}
</div>

</div>
""",
unsafe_allow_html=True
)

# =====================================================
# KPI ROW
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Model</div>
    <div class="metric-value">LSTM</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Vocabulary</div>
    <div class="metric-value">{len(tokenizer.word_index)}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Context Window</div>
    <div class="metric-value">{max_len}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">Status</div>
    <div class="metric-value">🟢Active</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================
# INPUT AREA
# =====================================================

st.markdown(
"""
<div class="editor-label">
✍️ Enter Your Text
</div>
""",
unsafe_allow_html=True
)

user_text = st.text_area(
    "",
    height=260,
    placeholder="Start typing your sentence here..."
)

predict = st.button("✨ Predict Next Word")

# =====================================================
# PREDICTION
# =====================================================

if predict:

    if user_text.strip():

        with st.spinner("Analyzing context..."):

            seq = tokenizer.texts_to_sequences(
                [user_text]
            )[0]

            seq = pad_sequences(
                [seq],
                maxlen=max_len - 1,
                padding="pre"
            )

            pred = model.predict(
                seq,
                verbose=0
            )

            pred_index = np.argmax(pred)

            next_word = ""

            for word, index in tokenizer.word_index.items():

                if index == pred_index:
                    next_word = word
                    break

            confidence = float(
                np.max(pred) * 100
            )

        st.balloons()

        st.markdown(
            f"""
            <div class="suggestion-box">
            ✨ Suggested Next Word:
            <b>{next_word}</b>
            &nbsp;&nbsp;&nbsp;
            <span class="confidence-badge">
            {confidence:.2f}% confidence
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(confidence / 100)

        st.session_state.history.append(
            f"{user_text[:40]}... ➜ {next_word}"
        )

    else:
        st.warning("Please enter some text.")

# =====================================================
# STATUS BAR
# =====================================================

st.markdown(
f"""
<div class="status-bar">

🟢 Status: Active

&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;

📚 Tokens: {len(tokenizer.word_index)}

&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;

🧠 Context Window: {max_len}

&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;

⚡ Mode: Auto Predict

</div>
""",
unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.write("")

# st.markdown(
# """
# <center>

# <p style="color:#64748B;">
# Built with ❤️ using Streamlit + TensorFlow
# </p>

# </center>
# """,
# unsafe_allow_html=True
# )
st.markdown("""
<div style="
color:#64748B;
font-size:18px;
margin-top:-10px;
margin-bottom:20px;
font-weight:500;
">
AI-powered Next Word Prediction using LSTM Deep Learning Architecture
</div>
""", unsafe_allow_html=True)

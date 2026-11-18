import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="İstanbul Su Paneli", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stTopNav"] {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        width: max-content;
        gap: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

sayfalar = [
    st.Page("Anasayfa.py", title="Anasayfa", default=True),
    st.Page("pages/1_Baraj_Doluluk_Tahmini.py", title="Baraj Doluluk Tahmini"),
    st.Page("pages/2_Tahmin_Ajanı.py", title="Tahmin Ajanı"),
]
st.navigation(sayfalar, position="top").run()

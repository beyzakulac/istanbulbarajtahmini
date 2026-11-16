"""Panel sayfalarına arka plan fotoğrafı koyar (yazılar okunsun diye üstüne yarı saydam bir zemin eklenir)."""
import base64
from pathlib import Path

import streamlit as st

GORSELLER = Path(__file__).resolve().parents[1] / "app" / "assets"


def _tema():
    """Açık/koyu tema bilgisi (eski Streamlit sürümlerinde bulunamazsa 'light' sayılır)."""
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def arka_plan(dosya_adi):
    yol = GORSELLER / dosya_adi
    if not yol.exists():  # görsel yoksa panel yine çalışsın
        return
    veri = base64.b64encode(yol.read_bytes()).decode()
    zemin = "rgba(14, 17, 23, 0.82)" if _tema() == "dark" else "rgba(255, 255, 255, 0.84)"
    st.markdown(
        f"""
        <style>
        [data-testid="stApp"] {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.25), rgba(0, 0, 0, 0.25)),
                              url("data:image/jpeg;base64,{veri}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"], [data-testid="stBottom"] > div {{ background: transparent; }}
        [data-testid="stMainBlockContainer"], .block-container {{
            background: {zemin};
            border-radius: 16px;
            padding: 2rem 2.5rem !important;
            margin-top: 3rem;
            margin-bottom: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

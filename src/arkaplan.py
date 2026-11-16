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


def _base64(dosya_adi):
    yol = GORSELLER / dosya_adi
    return base64.b64encode(yol.read_bytes()).decode() if yol.exists() else None


def yan_menu_arka_plan(dosya_adi="kopru_gece.jpg"):
    """Yan menüye (sidebar) gece köprü fotoğrafı + koyu saydam katman; yazılar beyaz."""
    veri = _base64(dosya_adi)
    if not veri:
        return
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-image: linear-gradient(rgba(8, 14, 30, 0.68), rgba(8, 14, 30, 0.68)),
                              url("data:image/jpeg;base64,{veri}");
            background-size: cover;
            background-position: center;
        }}
        [data-testid="stSidebar"] > div:first-child {{ background: transparent; }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] small,
        [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span {{
            color: #f5f7fa !important;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.7);
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNav"] a:hover {{
            background: rgba(255, 255, 255, 0.18) !important;
        }}
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] button {{
            color: #111 !important;
            text-shadow: none;
        }}
        [data-testid="stSidebar"] button p {{ color: #111 !important; text-shadow: none; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def arka_plan(dosya_adi):
    yan_menu_arka_plan()
    veri = _base64(dosya_adi)
    if not veri:  # görsel yoksa panel yine çalışsın
        return
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

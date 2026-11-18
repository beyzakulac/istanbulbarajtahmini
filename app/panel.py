"""Panelin giriş dosyası: sayfaları üstte yatay menü olarak gösterir.
Çalıştırma: streamlit run app/panel.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo kökü: src/ içe aktarılabilsin

st.set_page_config(page_title="İstanbul Su Paneli", page_icon="💧", layout="wide")

sayfalar = [
    st.Page("Anasayfa.py", title="Anasayfa", icon="🏠", default=True),
    st.Page("pages/1_Baraj_Doluluk_Tahmini.py", title="Baraj Doluluk Tahmini", icon="🌊"),
    st.Page("pages/2_Tahmin_Ajanı.py", title="Tahmin Ajanı", icon="🤖"),
]
st.navigation(sayfalar, position="top").run()

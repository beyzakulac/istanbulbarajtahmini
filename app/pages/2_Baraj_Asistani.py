"""Baraj Asistanı: tahmin modelini araç olarak kullanan yapay zekâ agent'ı (Sinem)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo kökü: src/ içe aktarılabilsin

import pandas as pd
import streamlit as st

from src.asistan import BarajAraclari, gemini_cevap, kural_tabanli_cevap

KOK = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Baraj Asistanı", page_icon="🤖", layout="wide")
st.title("🤖 Baraj Asistanı (AI Agent)")
st.markdown(
    "Sorunuzu Türkçe yazın. Asistan soruyu anlar, **hangi aracı kullanacağına kendisi karar verir** "
    "(doluluk verisi, 30 gün tahmini, erken uyarı, dönem istatistikleri, model başarısı), aracı çalıştırır ve "
    "sonucu açıklar. Veri **2011–2023** arasını kapsar."
)


@st.cache_resource
def araclari_hazirla():
    return BarajAraclari(str(KOK / "data" / "raw") + "/")


def anahtar_bul():
    """Gemini anahtarı: önce .streamlit/secrets.toml, sonra ortam değişkeni, sonra yan menü."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


with st.sidebar:
    st.subheader("Ayarlar")
    anahtar = anahtar_bul() or st.text_input("Gemini API anahtarı", type="password",
                                             help="aistudio.google.com adresinden ücretsiz alınır.")
    mod = st.radio("Çalışma modu", ["Gemini (LLM agent)", "Kural tabanlı (yedek)"],
                   index=0 if anahtar else 1)
    if st.button("Sohbeti temizle"):
        st.session_state.mesajlar = []
    st.caption("Örnek sorular:\n- 1 Mart 2023'ten 30 gün sonrası ne olur?\n- 15.07.2023 için %25 eşiğinde uyarı var mı?\n"
               "- 2021 yılının ortalama ve medyan doluluğu nedir?\n- Model ne kadar başarılı?\n"
               "- 2014 kuraklığını 2023 ile karşılaştır.")

araclar = araclari_hazirla()
st.session_state.setdefault("mesajlar", [])

for m in st.session_state.mesajlar:
    with st.chat_message("user" if m["rol"] == "user" else "assistant"):
        st.markdown(m["metin"])
        if m.get("adimlar"):
            with st.expander("Agent'ın çağırdığı araçlar"):
                st.json(m["adimlar"])

soru = st.chat_input("Örneğin: 1 Haziran 2023'ten 30 gün sonra doluluk ne olur?")
if soru:
    with st.chat_message("user"):
        st.markdown(soru)
    araclar.kayit = []
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor, araç çağırıyor..."):
            if mod.startswith("Gemini") and anahtar:
                gecmis = [{"rol": m["rol"], "metin": m["metin"]} for m in st.session_state.mesajlar[-10:]]
                try:
                    cevap, model = gemini_cevap(araclar, gecmis, soru, anahtar)
                    cevap = (cevap or "Cevap üretilemedi.") + f"\n\n_({model})_"
                except Exception as e:
                    araclar.kayit = []
                    cevap = (kural_tabanli_cevap(araclar, soru) +
                             f"\n\n_Gemini'ye ulaşılamadı, kural tabanlı yedek mod kullanıldı. ({str(e)[:120]})_")
            else:
                cevap = kural_tabanli_cevap(araclar, soru) + "\n\n_(kural tabanlı mod)_"
        st.markdown(cevap)
        adimlar = list(araclar.kayit)
        if adimlar:
            with st.expander("Agent'ın çağırdığı araçlar"):
                st.json(adimlar)
    st.session_state.mesajlar += [{"rol": "user", "metin": soru},
                                  {"rol": "model", "metin": cevap, "adimlar": adimlar}]

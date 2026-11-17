import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  

import numpy as np
import pandas as pd
import streamlit as st

from src.arkaplan import arka_plan
from src.grafik import cubuk_grafik, turkce_cizgi_grafik

KOK = Path(__file__).resolve().parents[1]

AD_DUZELT = {"B.ÇEKMECE": "BÜYÜKÇEKMECE", "K.ÇEKMECE": "KÜÇÜKÇEKMECE", "G.O.PAŞA": "GAZİOSMANPAŞA"}

st.set_page_config(page_title="İstanbul Su Paneli", layout="wide")
arka_plan("istanbul_bogaz.jpg")
st.title("İstanbul Baraj Doluluk ve Su Tüketimi Paneli")



@st.cache_data
def aylik_2023():
    """2023 aylık ilçe tüketimi (İBB 2023 dosyasında her ay ayrı sütun)."""
    aylik = pd.read_excel(KOK / "data" / "raw" / "ilce-baznda-su-tuketim-miktar-2023.xlsx")
    aylik.columns = [str(c).strip() for c in aylik.columns]
    aylik = aylik.dropna(subset=["İLÇELER"])
    aylik["İLÇELER"] = aylik["İLÇELER"].str.strip().replace(AD_DUZELT)
    aylik = aylik[aylik["İLÇELER"] != "GENEL TOPLAM"].set_index("İLÇELER")
    return aylik[[str(a) for a in range(202301, 202313)]]


@st.cache_data
def yillik_tuketim():
    """2011-2022 yıllık ilçe tüketimi (temizlenmiş veriden) + 2023 gerçek değerleri (aylık dosyanın toplamı)."""
    gunluk = pd.read_csv(KOK / "data" / "temizlenmis_gunluk_tuketim.csv", parse_dates=["Tarih"])
    yillik = gunluk.groupby(["İlçe", gunluk["Tarih"].dt.year])["Gunluk_Tuketim_m3"].sum().unstack()
    yillik.index = yillik.index.str.strip()
    yillik[2023] = aylik_2023().sum(axis=1)
    return yillik / 1e6   


yillik = yillik_tuketim()
aylik = aylik_2023() / 1e6
YILLAR = list(range(2011, 2023))



# ---  Seçilen ilçe için yıllık / aylık tüketim miktarı ---
st.header(" İlçe Bazındaki Tüketim Miktarı")
ilce = st.selectbox("İlçe seçin", sorted(yillik.index), index=sorted(yillik.index).index("ADALAR"))
gorunum = st.radio("Görünüm", ["Yıllık (2011–2023)", "Aylık (2023)"], horizontal=True)

if gorunum.startswith("Yıllık"):
    seri = yillik.loc[ilce, YILLAR + [2023]].rename("Tüketim (milyon m³)")
    seri.index = pd.to_datetime([f"{y}-01-01" for y in seri.index])
    seri.index.name = "Yıl"
    turkce_cizgi_grafik(seri, "Milyon m³", yillik=True)
else:
    seri = aylik.loc[ilce].rename("Tüketim (milyon m³)")
    seri.index = pd.to_datetime(seri.index, format="%Y%m")
    seri.index.name = "Ay"
    turkce_cizgi_grafik(seri, "Milyon m³", aylik=True)
    st.caption("Aylık veri İBB'de sadece 2023 için yayımlanıyor; 2011–2022 yılları yalnızca yıllık toplam olarak var.")

artis = (yillik.loc[ilce, 2022] / yillik.loc[ilce, 2011] - 1) * 100
c1, c2 = st.columns(2)
c1.metric("2022 tüketimi", f"{yillik.loc[ilce, 2022]:.2f} milyon m³")
c2.metric("2011 → 2022 değişim", f"%{artis:+.0f}")

# ---  En çok tüketim yapan ilçeler ---
st.header(" 2022'de en çok su tüketen ilçeler")
st.bar_chart(yillik[2022].sort_values(ascending=False).head(15), horizontal=True, x_label="Milyon m³")

# ---  2023 tahmini: trend modeli taban modeli geçebiliyor mu? ---
st.header(" 2023 Yılı İçin Tüketim Tahmini")

trend = yillik[YILLAR].apply(lambda s: np.polyval(np.polyfit(YILLAR, s.values, 1), 2023), axis=1)
gercek = yillik[2023]
mape = lambda t: (abs(t - gercek) / gercek).mean() * 100
sonuc = pd.DataFrame({"Ortalama yüzde hata (MAPE)": [mape(yillik[2022]), mape(trend)]},
                     index=["Taban: 2023 = 2022", "Doğrusal trend"])
st.dataframe(sonuc.style.format("%{:.1f}"), width="content")
cubuk_grafik(sonuc["Ortalama yüzde hata (MAPE)"], "Ortalama yüzde hata (%)", yatay=True, bicim=".1f", onek="%")

detay = pd.DataFrame({"2022": yillik[2022], "Gerçek 2023": gercek, "Taban tahmini": yillik[2022], "Trend tahmini": trend})
st.dataframe(detay.loc[[ilce]].style.format("{:.2f}"), width="stretch")
cubuk_grafik(detay.loc[ilce], "Tüketim (milyon m³)", vurgula="Gerçek 2023")

st.caption("Veri seti İBB Açık Veri Portalı'ndan alınmıştır.")

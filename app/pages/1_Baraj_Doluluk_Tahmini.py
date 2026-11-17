import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from src.degerlendir import UFUK, bol, metrikler
from src.ozellikler import model_verisi, modeller
from src.veri import birlesik_veri
from src.arkaplan import arka_plan
from src.grafik import turkce_cizgi_grafik

KOK = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Baraj Doluluk Tahmini", layout="wide")
arka_plan("baraj.jpg")
st.title("İstanbul Baraj Doluluk Tahmini ve Erken Uyarı Sistemi")


@st.cache_data
def veri_getir():
    df = birlesik_veri(str(KOK / "data" / "raw") + "/")
    return df, bol(model_verisi(df))


@st.cache_resource
def modelleri_egit():
    _, p = veri_getir()
    oz = [c for c in p["egitim_tum"].columns if c != "hedef"]
    egitilmis = {}
    for ad, m in modeller().items():
        m.fit(p["egitim_tum"][oz], p["egitim_tum"]["hedef"])
        egitilmis[ad] = m
    return egitilmis, oz


df, p = veri_getir()
egitilmis, oz = modelleri_egit()
test = p["test"]
gercek = test["doluluk"] + test["hedef"]
hedef_tarih = test.index + pd.Timedelta(days=UFUK)

# --- Model başarısı ---
st.header("2023 Yılı İçin Model Başarısı")
mevsim = p["egitim_tum"].groupby(p["egitim_tum"].index.dayofyear)["hedef"].mean()
tahminler = {
    "Taban: doluluk değişmez": test["doluluk"].values,
    "Taban: mevsimsel ortalama": test["doluluk"].values + test.index.dayofyear.map(mevsim).fillna(0).values,
}
for ad, m in egitilmis.items():
    tahminler[ad] = test["doluluk"].values + m.predict(test[oz])

tablo = pd.DataFrame({ad: metrikler(gercek, t) for ad, t in tahminler.items()}).T
tablo.columns = ["MAE", "RMSE", "R²"]
st.dataframe(tablo.style.format("{:.2f}").highlight_min(subset=["MAE"], color="#c6efce"),
             width="stretch")
st.subheader("Modellerin 2023 hatası (MAE, düşük daha iyi)")
st.bar_chart(tablo["MAE"].sort_values(), horizontal=True,
             x_label="MAE (yüzde puan)", y_label="Model")

lstm_dosya = KOK / "sonuclar" / "karsilastirma_2023.csv"
if lstm_dosya.exists():
    st.subheader("LSTM dahil tüm modeller")
    st.dataframe(pd.read_csv(lstm_dosya, index_col=0).style.format("{:.2f}"), width="stretch")

secilen = st.multiselect("Grafikte gösterilecek tahminler", list(tahminler),
                         default=["Taban: doluluk değişmez", "Doğrusal Regresyon", "XGBoost"])
grafik = pd.DataFrame({"Gerçek": gercek.values, **{ad: tahminler[ad] for ad in secilen}}, index=hedef_tarih)
grafik.index.name = "Tarih"
turkce_cizgi_grafik(grafik, "Toplam doluluk (%)")


# --- Erken uyarı ---
st.header("Erken Uyarı Sistemi")
col1, col2 = st.columns(2)
esik = col1.slider("Uyarı eşiği (%)", 10, 60, 30)
model_adi = col2.selectbox("Model", list(egitilmis), index=0)
tarih = st.date_input(
    "Bugünün verisiyle 30 gün sonrası için tahmin edilir.",
    value=test.index[-1].date(), min_value=test.index[0].date(), max_value=test.index[-1].date(),
)
satir = test.loc[[pd.Timestamp(tarih)]]
bugun = float(satir["doluluk"].iloc[0])
tahmin = bugun + float(egitilmis[model_adi].predict(satir[oz])[0])
gerceklesen = bugun + float(satir["hedef"].iloc[0])

c1, c2, c3 = st.columns(3)
c1.metric("Bugünkü doluluk", f"%{bugun:.1f}")
c2.metric(f"{UFUK} gün sonrası tahmin", f"%{tahmin:.1f}", f"{tahmin - bugun:+.1f} puan")
c3.metric("Gerçekleşen (kontrol için)", f"%{gerceklesen:.1f}")
if tahmin < esik:
    st.error(f"UYARI: {UFUK} gün sonra doluluğun %{esik} eşiğinin altına düşmesi bekleniyor (%{tahmin:.1f}).")
else:
    st.success(f"{UFUK} gün sonra doluluğun %{esik} eşiğinin üzerinde kalması bekleniyor (%{tahmin:.1f}).")

# --- XGBoost özellik önemleri ---
st.header("XGBoost Özellik Önemleri")
onem = pd.Series(egitilmis["XGBoost"].feature_importances_, index=oz).sort_values(ascending=False)
st.bar_chart(onem, horizontal=True, x_label="Önem")

st.caption("Veri seti İBB Açık Veri Portalı'ndan alınmıştır.")
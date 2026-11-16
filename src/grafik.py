"""Tarih eksenli grafikleri Türkçe ay/gün adlarıyla çizer (Streamlit'in varsayılanı İngilizce)."""
import altair as alt
import pandas as pd
import streamlit as st

TURKCE_ZAMAN = {
    "dateTime": "%e %B %Y %A %X",
    "date": "%d.%m.%Y",
    "time": "%H:%M:%S",
    "periods": ["ÖÖ", "ÖS"],
    "days": ["Pazar", "Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi"],
    "shortDays": ["Paz", "Pzt", "Sal", "Çar", "Per", "Cum", "Cmt"],
    "months": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
               "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
    "shortMonths": ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"],
}


def turkce_cizgi_grafik(tablo, y_etiketi):
    """st.line_chart yerine: index tarih, her sütun bir çizgi; eksen ve ipucu Türkçe."""
    uzun = tablo.reset_index().melt(id_vars=tablo.index.name or "index", var_name="Seri", value_name="Değer")
    tarih_sutunu = tablo.index.name or "index"
    grafik = (
        alt.Chart(uzun)
        .mark_line()
        .encode(
            x=alt.X(f"{tarih_sutunu}:T", title=None, axis=alt.Axis(format="%b %Y", tickCount="month")),
            y=alt.Y("Değer:Q", title=y_etiketi),
            color=alt.Color("Seri:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip(f"{tarih_sutunu}:T", title="Tarih", format="%d %B %Y"),
                     alt.Tooltip("Seri:N", title="Seri"),
                     alt.Tooltip("Değer:Q", title=y_etiketi, format=".2f")],
        )
        .configure(locale={"time": TURKCE_ZAMAN})
    )
    st.altair_chart(grafik, use_container_width=True)

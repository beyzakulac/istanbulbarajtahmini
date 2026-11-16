"""Baraj Asistanı: tahmin modelini ve veriyi araç (tool) olarak kullanan yapay zekâ agent'ı.

Nasıl çalışır?
1. Kullanıcı soruyu Türkçe yazar ("1 Mart 2023'te 30 gün sonrası ne olur?").
2. Dil modeli (Gemini) soruyu okur ve hangi aracı, hangi parametreyle çağıracağına KENDİSİ karar verir.
3. Araç bizim Python kodumuzdur (XGBoost tahmini, veri özeti...). Sonucu modele geri döner.
4. Model gerekirse başka araç da çağırır, sonra sonucu Türkçe açıklar.

Gemini anahtarı yoksa ya da internet yoksa 'kural tabanlı yedek mod' devreye girer:
sorudaki kelimelere ve tarihe bakarak aynı araçları çağırır.
"""
import re

import pandas as pd

from src.degerlendir import UFUK, bol, metrikler
from src.ozellikler import model_verisi, modeller
from src.veri import KAPASITE, birlesik_veri, doluluk_oku

ILK_TARIH = "2011-01-01"
SON_TARIH = "2023-12-31"

# Sırayla denenir; ilk çalışan kullanılır (Google model adlarını zamanla değiştiriyor).
GEMINI_MODELLERI = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-2.0-flash"]

SISTEM_TALIMATI = f"""Sen 'Baraj Asistanı'sın: İstanbul barajlarının doluluğu hakkında soruları cevaplayan bir yapay zekâ agent'ısın.
Kurallar:
- Sayıları ASLA kendin uydurma; her sayıyı araçlardan al. Gerekirse birden fazla araç çağır.
- Veri {ILK_TARIH} ile {SON_TARIH} arasını kapsar. Bu aralık dışındaki tarihler için bunu kibarca söyle.
- Tahmin {UFUK} gün sonrası içindir. Modeller 2011-2022 ile eğitildi, 2023 hiç görülmemiş test yılıdır.
- Tarih belirtilmezse {SON_TARIH[:4]} yılının son verili gününü (araçların varsayılanı) kullan.
- Cevabı kısa, anlaşılır Türkçe ile ver; doluluğu yüzde (%) olarak yaz, hangi aracı kullandığını tek cümleyle belirt.
- Su yönetimi önerisi istenirse genel ve sorumlu öneriler ver (tasarruf, izleme), resmi karar gibi sunma.
"""


def _tarih(t, varsayilan):
    """'2023-03-01', '01.03.2023' gibi tarihleri Timestamp'e çevirir."""
    if not t:
        return pd.Timestamp(varsayilan)
    t = str(t).strip()
    if re.fullmatch(r"\d{1,2}[./]\d{1,2}[./]\d{4}", t):
        return pd.to_datetime(t.replace("/", "."), format="%d.%m.%Y")
    return pd.Timestamp(t)


class BarajAraclari:
    """Agent'ın kullanabildiği araçlar. Her araç sözlük (dict) döndürür."""

    def __init__(self, klasor):
        self.df = birlesik_veri(klasor, bitis=SON_TARIH)
        self.barajlar = doluluk_oku(klasor)[list(KAPASITE)][ILK_TARIH:SON_TARIH] * 100
        self.veri = model_verisi(self.df)
        p = bol(self.veri)
        self.oz = [c for c in self.veri.columns if c != "hedef"]
        self.modeller = {}
        for ad, m in modeller().items():           # 2011-2022 ile eğit (panelle aynı)
            m.fit(p["egitim_tum"][self.oz], p["egitim_tum"]["hedef"])
            self.modeller[ad] = m
        self.test = p["test"]
        self.kayit = []                             # demoda "agent hangi aracı çağırdı" göstermek için

    # ---------- yardımcı ----------
    def _gun(self, tarih, tablo):
        t = _tarih(tarih, tablo.index[-1])
        if t < tablo.index[0] or t > tablo.index[-1]:
            raise ValueError(f"{t.date()} veri aralığı dışında ({tablo.index[0].date()} - {tablo.index[-1].date()}).")
        return t

    def _kaydet(self, ad, girdi, sonuc):
        self.kayit.append({"araç": ad, "girdi": girdi, "sonuç": sonuc})
        return sonuc

    # ---------- araçlar ----------
    def doluluk_getir(self, tarih: str = "") -> dict:
        """Verilen tarihteki İstanbul toplam baraj doluluğunu (%) ve baraj baraj doluluğu getirir.

        Args:
            tarih: 'YYYY-AA-GG' biçiminde tarih (2011-2023). Boş bırakılırsa son verili gün.
        """
        try:
            t = self._gun(tarih, self.df)
            sonuc = {"tarih": str(t.date()), "toplam_doluluk_yuzde": round(float(self.df.loc[t, "toplam"]), 1),
                     "barajlar_yuzde": self.barajlar.loc[t].round(1).to_dict()}
        except Exception as e:
            sonuc = {"hata": str(e)}
        return self._kaydet("doluluk_getir", {"tarih": tarih}, sonuc)

    def tahmin_yap(self, tarih: str = "", model: str = "XGBoost") -> dict:
        """Verilen günün verisiyle 30 gün sonraki toplam doluluğu tahmin eder (XGBoost veya Doğrusal Regresyon).
        Tarih geçmişteyse gerçekleşen değeri de karşılaştırma için verir.

        Args:
            tarih: Tahminin yapıldığı gün, 'YYYY-AA-GG' (2011-12-01 ile 2023-12-01 arası). Boş = son mümkün gün.
            model: 'XGBoost' veya 'Doğrusal Regresyon'.
        """
        try:
            if model not in self.modeller:
                model = "XGBoost"
            t = self._gun(tarih, self.veri)
            satir = self.veri.loc[[t]]
            bugun = float(satir["doluluk"].iloc[0])
            tahmin = bugun + float(self.modeller[model].predict(satir[self.oz])[0])
            sonuc = {"tahmin_gunu": str(t.date()), "hedef_gun": str((t + pd.Timedelta(days=UFUK)).date()),
                     "model": model, "bugunku_doluluk": round(bugun, 1), "tahmin": round(tahmin, 1),
                     "degisim_puan": round(tahmin - bugun, 1),
                     "gerceklesen": round(bugun + float(satir["hedef"].iloc[0]), 1),
                     "test_yili_mi": bool(t.year == 2023),
                     "son_30_gun_yagis_mm": round(float(satir["yagis_30"].iloc[0]), 1)}
        except Exception as e:
            sonuc = {"hata": str(e)}
        return self._kaydet("tahmin_yap", {"tarih": tarih, "model": model}, sonuc)

    def uyari_kontrol(self, tarih: str = "", esik: float = 30.0) -> dict:
        """30 gün sonraki tahmin doluluğun verilen eşiğin (%) altına düşüp düşmeyeceğini kontrol eder.

        Args:
            tarih: Tahminin yapıldığı gün, 'YYYY-AA-GG'. Boş = son mümkün gün.
            esik: Uyarı eşiği, yüzde olarak (örneğin 30).
        """
        t = self.tahmin_yap(tarih)
        self.kayit.pop()                            # iç çağrıyı ayrıca gösterme
        if "hata" in t:
            return self._kaydet("uyari_kontrol", {"tarih": tarih, "esik": esik}, t)
        sonuc = {**t, "esik": esik, "uyari": bool(t["tahmin"] < esik),
                 "durum": "UYARI: eşiğin altına düşmesi bekleniyor" if t["tahmin"] < esik else "Normal: eşiğin üzerinde"}
        return self._kaydet("uyari_kontrol", {"tarih": tarih, "esik": esik}, sonuc)

    def donem_ozeti(self, baslangic: str = "2023-01-01", bitis: str = "2023-12-31") -> dict:
        """İki tarih arasındaki dönem için tanımlayıcı istatistikler: doluluk ortalama/medyan/en düşük/en yüksek,
        toplam yağış ve ortalama günlük su tüketimi.

        Args:
            baslangic: Dönemin ilk günü, 'YYYY-AA-GG'.
            bitis: Dönemin son günü, 'YYYY-AA-GG'.
        """
        try:
            b, s = _tarih(baslangic, ILK_TARIH), _tarih(bitis, SON_TARIH)
            d = self.df[b:s]
            if d.empty:
                raise ValueError("Bu aralıkta veri yok (veri 2011-2023 arası).")
            dol = d["toplam"]
            sonuc = {"baslangic": str(d.index[0].date()), "bitis": str(d.index[-1].date()), "gun_sayisi": len(d),
                     "doluluk_ortalama": round(float(dol.mean()), 1), "doluluk_medyan": round(float(dol.median()), 1),
                     "doluluk_en_dusuk": round(float(dol.min()), 1), "en_dusuk_tarih": str(dol.idxmin().date()),
                     "doluluk_en_yuksek": round(float(dol.max()), 1), "en_yuksek_tarih": str(dol.idxmax().date()),
                     "toplam_yagis_mm": round(float(d["yagis"].sum()), 1),
                     "ortalama_gunluk_tuketim_milyon_m3": round(float(d["tuketim"].mean()) / 1e6, 2)}
        except Exception as e:
            sonuc = {"hata": str(e)}
        return self._kaydet("donem_ozeti", {"baslangic": baslangic, "bitis": bitis}, sonuc)

    def model_basarisi(self) -> dict:
        """Modellerin 2023 test yılındaki başarısını (MAE, RMSE, R²) ve 'doluluk değişmez' taban çizgisini verir."""
        gercek = self.test["doluluk"] + self.test["hedef"]
        tah = {"Taban: doluluk değişmez": self.test["doluluk"]}
        for ad, m in self.modeller.items():
            tah[ad] = self.test["doluluk"] + m.predict(self.test[self.oz])
        sonuc = {ad: {k: round(float(v), 2) for k, v in metrikler(gercek, t).items()} for ad, t in tah.items()}
        sonuc["aciklama"] = "MAE/RMSE doluluk yüzde puanıdır, düşük olan iyidir. R² 1'e yakınsa iyidir."
        return self._kaydet("model_basarisi", {}, sonuc)

    def araclar(self):
        return [self.doluluk_getir, self.tahmin_yap, self.uyari_kontrol, self.donem_ozeti, self.model_basarisi]


# ---------------------------------------------------------------- Gemini agent
def gemini_cevap(araclar, gecmis, soru, anahtar):
    """Gemini'ye soruyu ve araçları verir; model araçları kendisi seçip çağırır (function calling).
    gecmis: [{'rol': 'user'|'model', 'metin': ...}] önceki konuşma. Döner: (cevap, kullanılan model adı)."""
    from google import genai
    from google.genai import types

    istemci = genai.Client(api_key=anahtar)
    icerik = [types.Content(role=m["rol"], parts=[types.Part(text=m["metin"])]) for m in gecmis]
    icerik.append(types.Content(role="user", parts=[types.Part(text=soru)]))
    ayar = types.GenerateContentConfig(
        system_instruction=SISTEM_TALIMATI, tools=araclar.araclar(), temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=6))
    son_hata = None
    for model in GEMINI_MODELLERI:
        try:
            yanit = istemci.models.generate_content(model=model, contents=icerik, config=ayar)
            return yanit.text, model
        except Exception as e:                       # model adı yoksa / kota dolduysa sıradakini dene
            son_hata = e
    raise RuntimeError(f"Gemini'ye ulaşılamadı: {son_hata}")


# ---------------------------------------------------------------- kural tabanlı yedek
AYLAR = {"ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "mayis": 5, "haziran": 6,
         "temmuz": 7, "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9, "ekim": 10, "kasım": 11, "kasim": 11,
         "aralık": 12, "aralik": 12}


def tarihleri_bul(metin):
    """Metindeki tarihleri bulur: 01.03.2023, 2023-03-01, '1 Mart 2023'."""
    m = metin.lower()
    bulunan = re.findall(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4})\b", m)
    for gun, ay, yil in re.findall(r"\b(\d{1,2})\s+([a-zçğıöşü]+)\s+(\d{4})\b", m):
        if ay in AYLAR:
            bulunan.append(f"{yil}-{AYLAR[ay]:02d}-{int(gun):02d}")
    return [str(_tarih(t, SON_TARIH).date()) for t in bulunan]


def kural_tabanli_cevap(araclar, soru):
    """LLM olmadan: anahtar kelimeye göre araç seçer, sonucu şablonla yazar."""
    s = soru.lower()
    tarihler = tarihleri_bul(soru)
    yil = re.search(r"\b(20[12]\d)\b", s)
    esik = re.search(r"%\s*(\d+)|(\d+)\s*%", s)

    if any(k in s for k in ["başarı", "basari", "mae", "rmse", "hata", "performans", "model ne kadar"]):
        r = araclar.model_basarisi()
        satirlar = [f"- **{ad}**: MAE {v['MAE']}, RMSE {v['RMSE']}, R² {v['R2']}" for ad, v in r.items() if ad != "aciklama"]
        return "2023 test yılı sonuçları (30 gün sonrası):\n" + "\n".join(satirlar) + f"\n\n{r['aciklama']}"

    if any(k in s for k in ["ortalama", "medyan", "özet", "ozet", "istatistik", "en düşük", "en yüksek"]):
        if len(tarihler) >= 2:
            b, e = tarihler[0], tarihler[1]
        elif yil:
            b, e = f"{yil.group(1)}-01-01", f"{yil.group(1)}-12-31"
        else:
            b, e = "2023-01-01", "2023-12-31"
        r = araclar.donem_ozeti(b, e)
        if "hata" in r:
            return f"Üzgünüm: {r['hata']}"
        return (f"**{r['baslangic']} – {r['bitis']}** ({r['gun_sayisi']} gün): doluluk ortalaması %{r['doluluk_ortalama']}, "
                f"medyanı %{r['doluluk_medyan']}. En düşük %{r['doluluk_en_dusuk']} ({r['en_dusuk_tarih']}), "
                f"en yüksek %{r['doluluk_en_yuksek']} ({r['en_yuksek_tarih']}). Toplam yağış {r['toplam_yagis_mm']} mm, "
                f"ortalama günlük tüketim {r['ortalama_gunluk_tuketim_milyon_m3']} milyon m³.")

    tarih = tarihler[0] if tarihler else ""
    if any(k in s for k in ["uyarı", "uyari", "eşik", "esik", "kritik", "tehlike"]):
        e = float(next(g for g in esik.groups() if g)) if esik else 30.0
        r = araclar.uyari_kontrol(tarih, e)
        if "hata" in r:
            return f"Üzgünüm: {r['hata']}"
        isaret = "⚠️" if r["uyari"] else "✅"
        return (f"{isaret} {r['tahmin_gunu']} verisine göre {r['hedef_gun']} için tahmin %{r['tahmin']} "
                f"(eşik %{r['esik']}). Durum: {r['durum']}. Gerçekleşen: %{r['gerceklesen']}.")

    if any(k in s for k in ["tahmin", "sonra", "olacak", "ne olur", "gelecek"]):
        model = "Doğrusal Regresyon" if ("doğrusal" in s or "dogrusal" in s or "regresyon" in s) else "XGBoost"
        r = araclar.tahmin_yap(tarih, model)
        if "hata" in r:
            return f"Üzgünüm: {r['hata']}"
        return (f"{r['model']} modeline göre {r['tahmin_gunu']} tarihinde %{r['bugunku_doluluk']} olan doluluğun "
                f"{r['hedef_gun']} tarihinde **%{r['tahmin']}** olması bekleniyor ({r['degisim_puan']:+} puan). "
                f"Gerçekleşen değer %{r['gerceklesen']} idi. Son 30 günde {r['son_30_gun_yagis_mm']} mm yağış vardı.")

    r = araclar.doluluk_getir(tarih)
    if "hata" in r:
        return f"Üzgünüm: {r['hata']}"
    en_dolu = max(r["barajlar_yuzde"], key=r["barajlar_yuzde"].get)
    en_bos = min(r["barajlar_yuzde"], key=r["barajlar_yuzde"].get)
    return (f"{r['tarih']} tarihinde İstanbul barajlarının toplam doluluğu **%{r['toplam_doluluk_yuzde']}**. "
            f"En dolu baraj {en_dolu} (%{r['barajlar_yuzde'][en_dolu]}), en boş baraj {en_bos} (%{r['barajlar_yuzde'][en_bos]}).")

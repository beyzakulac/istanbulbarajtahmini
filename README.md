# İstanbul Baraj Doluluk Tahmini

Yapay Zeka dersi projesi: İstanbul barajlarının toplam doluluk oranını 30 gün sonrası için tahmin etmek ve erken uyarı vermek.

- Veri: İBB Açık Veri Portalı (`data/raw/`)
- Bölme: eğitim 2011-2022, test 2023
- Metrikler: MAE, RMSE, R² (doluluk yüzde puanı)

## Kurulum
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Klasörler
- `data/raw/` ham Excel dosyaları (değiştirilmez)
- `data/processed/` temizlenmiş veri (git'e girmez, koddan üretilir)
- `notebooks/` keşif ve deneyler
- `src/` ortak kullanılan kod (temizleme, öznitelikler, değerlendirme)
- `app/` Streamlit paneli

## Panel
Çalıştırma: `streamlit run app/panel.py` (üstteki menüden sayfalar seçilir).
- **Anasayfa**: baraj doluluğu, yağış, tüketim ve ilçe bazında tüketim analizi
- **Baraj Doluluk Tahmini**: model başarısı (2023) ve erken uyarı sistemi (İstanbul toplamı ya da tek baraj)
- **Tahmin Ajanı**: Türkçe soruları cevaplayan yapay zekâ agent'ı

## Tahmin Ajanı (AI agent)
`app/pages/2_Tahmin_Ajanı.py` + `src/asistan.py`. Gemini dil modeli soruyu okur, hangi aracı
(doluluk_getir, tahmin_yap, uyari_kontrol, donem_ozeti, model_basarisi) çağıracağına kendisi karar verir
(function calling), sonucu Türkçe açıklar. Tahminler İstanbul toplamı ya da tek bir baraj için, 1-90 gün sonrası için yapılabilir. Anahtar yoksa kural tabanlı yedek mod çalışır.

Gemini anahtarı: https://aistudio.google.com/apikey adresinden ücretsiz alınır ve
`.streamlit/secrets.toml` dosyasına yazılır (bu dosya git'e girmez):
```toml
GEMINI_API_KEY = "buraya-anahtar"
```
Çalıştırma: `streamlit run app/panel.py` → üstteki menüden **Tahmin Ajanı**.

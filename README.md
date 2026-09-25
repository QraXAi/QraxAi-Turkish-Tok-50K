# QraXAi Turkish Tokenizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/coderian/QraXAi-Turkish-Tok-50K)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Türkçe metinler için sıfırdan eğitilmiş, **50.000 kelime dağarcıklı** BPE (Byte-Pair Encoding) tokenizer'ı. Bu depo, tokenizer'ı indiren, eğiten ve Hugging Face Hub'a yükleyen tüm betikleri içerir.

Eğitilmiş tokenizer: **[coderian/QraXAi-Turkish-Tok-50K](https://huggingface.co/coderian/QraXAi-Turkish-Tok-50K)**

## Özellikler

- **Model tipi:** BPE (Byte-Pair Encoding)
- **Kelime dağarcığı:** 50.000 token (46.073 merge kuralı)
- **Normalizasyon:** NFC (Unicode Normalization Form C)
- **Ön tokenizasyon:** Whitespace
- **Özel tokenlar:** `[PAD]` (0), `[UNK]` (1), `[BOS]` (2), `[EOS]` (3)
- **Eğitim verisi:** `tascib/turkish-llm-dataset` — ~5,7 GB, 2.663.578 satır, 721.478.261 kelime
- **Uyumluluk:** `transformers` ve `tokenizers` kütüphaneleriyle doğrudan kullanılabilir

## Proje Yapısı

```
.
├── download.py                 # Derlemi indirir (streaming)
├── train.py                    # Tokenizer'ı eğitir ve Hub'a yükler
├── count.cpp                   # Derlem kelime/satır sayacı (C++)
├── QraXAi-Turkish-Tok-50K/     # Eğitilmiş tokenizer dosyaları
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── README.md               # Hugging Face model kartı
├── LICENSE
└── .gitignore
```

> Not: `corpus.txt` (5,7 GB) ve `venv/` depoya dahil edilmez; aşağıdaki adımlarla yerel olarak oluşturulur.

## Kurulum

Python 3.11+ önerilir.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install datasets transformers tokenizers huggingface_hub
```

## Kullanım

### 1. Derlemi indir

```bash
python download.py
```

Veri kümesi akış (streaming) modunda indirilir ve `corpus.txt` dosyasına yazılır.

### 2. Kelime sayımı (isteğe bağlı)

```bash
g++ -O2 -o count count.cpp
./count
```

Örnek çıktı:

```
Kelime sayisi : 721478261
Satir sayisi  : 2663578
```

### 3. Tokenizer'ı eğit

```bash
python train.py
```

Betik tokenizer'ı eğitir, `QraXAi-Turkish-Tok-50K/` dizinine kaydeder, örnek metinlerle test eder ve Hugging Face Hub'a yükler. Hub'a yüklemek için önceden giriş yapmış olmanız gerekir:

```bash
hf auth login
```

### 4. Eğitilmiş tokenizer'ı kullan

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("coderian/QraXAi-Turkish-Tok-50K")

metin = "Türkçe doğal dil işleme çalışmaları hızla ilerliyor."
encoded = tokenizer(metin, add_special_tokens=False)

print(encoded["input_ids"])
# [7387, 5629, 4714, 9882, 6209, 7714, 19660, 21]

print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
# ['Türkçe', 'doğal', 'dil', 'işleme', 'çalışmaları', 'hızla', 'ilerliyor', '.']
```

Yerel dosyalardan yüklemek için:

```python
tokenizer = AutoTokenizer.from_pretrained("QraXAi-Turkish-Tok-50K")
```

## Eğitim Ayarları

`train.py` içindeki temel parametreler:

| Parametre | Değer |
|-----------|-------|
| `VOCAB_SIZE` | 50.000 |
| `MIN_FREQUENCY` | 2 |
| Normalizer | `NFC` |
| Pre-tokenizer | `Whitespace` |
| Özel tokenlar | `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]` |

## Bilinen Sınırlamalar

- **Boşluk kaybı:** `Whitespace` ön tokenizasyon nedeniyle `decode` çıktısında noktalama işaretlerinden önce fazladan boşluk görünebilir (`"Merhaba dünya!"` → `Merhaba dünya !`).
- **Byte-level geri dönüş yok:** Dağarcık dışındaki karakterler `[UNK]` olarak işaretlenir.
- **Büyük/küçük harf duyarlılığı:** Lowercase uygulanmaz; `Ankara` ve `ankara` farklı token dizileri üretir.

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

Eğitim verisi (`tascib/turkish-llm-dataset`) kendi lisans koşullarına tabidir.

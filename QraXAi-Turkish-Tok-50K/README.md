---
language:
- tr
library_name: tokenizers
tags:
- tokenizer
- bpe
- turkish
- turkce
---

# QraXAi Turkish Tokenizer 50K

Türkçe metinler için sıfırdan eğitilmiş, **50.000 kelime dağarcıklı** BPE (Byte-Pair Encoding) tokenizer'ı.

Türkçenin eklemeli yapısına ve yoğun çekimli kelime formlarına uygun olarak, gerçek Türkçe metinlerden oluşan yaklaşık **5,7 GB**'lık bir derlem üzerinde eğitilmiştir.

## Özellikler

- **Model tipi:** BPE (Byte-Pair Encoding)
- **Kelime dağarcığı:** 50.000 token
- **Birleştirme (merge) kuralı:** 46.073
- **Normalizasyon:** NFC (Unicode Normalization Form C)
- **Ön tokenizasyon:** Whitespace
- **Minimum frekans:** 2
- **Dosya formatı:** `tokenizers` (hızlı tokenizer), `transformers` ile uyumlu

## Özel Tokenlar

| Token | ID | Açıklama |
|-------|----|----------|
| `[PAD]` | 0 | Dolgu |
| `[UNK]` | 1 | Bilinmeyen |
| `[BOS]` | 2 | Dizinin başlangıcı |
| `[EOS]` | 3 | Dizinin sonu |

## Eğitim Verisi

Tokenizasyon aracı, `tascib/turkish-llm-dataset` veri kümesinden akış (streaming) modunda indirilen Türkçe metinlerle eğitilmiştir.

| Ölçüt | Değer |
|------------|----------------|
| Kaynak | `tascib/turkish-llm-dataset` |
| Dosya boyutu | ~5,7 GB |
| Satır sayısı | 2.663.578 |
| Kelime sayısı | 721.478.261 |

## Kullanım

Kurulum:

```bash
pip install transformers tokenizers
```

### transformers ile

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

### tokenizers ile

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("coderian/QraXAi-Turkish-Tok-50K")

encoded = tokenizer.encode("Merhaba dünya!")
print(encoded.tokens)
# ['Merhaba', 'dünya', '!']
print(encoded.ids)
```

## Tokenizasyon Örnekleri

| Metin | Token sayısı | Tokenlar |
|-------|--------------|----------|
| `Merhaba dünya!` | 3 | `['Merhaba', 'dünya', '!']` |
| `Türkçe doğal dil işleme çalışmaları hızla ilerliyor.` | 8 | `['Türkçe', 'doğal', 'dil', 'işleme', 'çalışmaları', 'hızla', 'ilerliyor', '.']` |
| `Işık, ışıkta ve ışıltılı bir şekilde parlıyor.` | 10 | `['Işık', ',', 'ışıkta', 've', 'ışıltılı', 'bir', 'şekilde', 'par', 'lıyor', '.']` |
| `Şimdi öğrenmeye başlıyoruz, çalışmalarımızı bugün tamamladık.` | 8 | `['Şimdi', 'öğrenmeye', 'başlıyoruz', ',', 'çalışmalarımızı', 'bugün', 'tamamladık', '.']` |

## Verimlilik

Türkçe derlemden alınan rastgele 200 satırlık bir örneklem üzerinde ölçülen değerler:

| Ölçüt | Değer |
|-----------------|-------|
| Token / kelime | ~1,44 |
| Karakter / token | ~5,76 |

Türkçe karakterler (`ç, ğ, ı, İ, ö, ş, ü` ve büyük/küçük biçimleri) ve yaygın Türkçe ekler kelime dağarcığında temsil edilmektedir; bu sayede çekimli kelimeler genellikle birkaç token'a ayrılır.

## Bilinen Sınırlamalar

- **Boşluk kaybı (Whitespace ön tokenizasyon):** Çözme (`decode`) işlemi sırasında token'lar boşlukla birleştirildiğinden, noktalama işaretlerinden önce fazladan boşluk görünebilir. Örnek: `"Merhaba dünya!"` → `Merhaba dünya !`. Bu davranış `Whitespace` ön tokenizasyonun doğal bir sonucudur.
- **Byte-level geri dönüş yok:** Model byte-level BPE kullanmaz; dağarcık dışında kalan karakterler `[UNK]` olarak işaretlenir. Günlük Türkçe metinlerde bu durumla nadiren karşılaşılır.
- **Büyük/küçük harf duyarlılığı:** Tokenizer küçük harfe çevirme (lowercase) uygulamaz; `Ankara` ve `ankara` farklı token dizileri üretir.
- **Unicode NFC:** Metinler NFC biçimine normalize edilir. Farklı bir normalizasyon biçiminde (ör. NFD) yazılmış girdiler eğitim dağılımından sapabilir.

## Yeniden Üretim

```bash
# 1) Derlemi indir
python download.py

# 2) Tokenizer'ı eğit ve yükle
python train.py
```

Eğitim betiği (`train.py`) şu ayarları kullanır:

```python
VOCAB_SIZE = 50_000
MIN_FREQUENCY = 2

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
)
```

## Atıf

```bibtex
@misc{qraxai-turkish-tok-50k,
  title  = {QraXAi Turkish Tokenizer 50K},
  author = {QraXAi},
  year   = {2026},
  url    = {https://huggingface.co/coderian/QraXAi-Turkish-Tok-50K}
}
```

---

## English Summary

**QraXAi Turkish Tokenizer 50K** is a BPE tokenizer with a 50,000-token vocabulary, trained from scratch on ~5.7 GB of Turkish text (721M words, ~2.66M documents) from the `tascib/turkish-llm-dataset`. It uses NFC normalization, whitespace pre-tokenization, and the special tokens `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("coderian/QraXAi-Turkish-Tok-50K")
```

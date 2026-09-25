from pathlib import Path

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


CORPUS_FILE = "corpus.txt"
OUTPUT_DIR = "QraXAi-Turkish-Tok-50K"
HF_REPO = "coderian/QraXAi-Turkish-Tok-50K"

VOCAB_SIZE = 50_000
MIN_FREQUENCY = 2


def main():
    corpus = Path(CORPUS_FILE)

    if not corpus.exists():
        raise FileNotFoundError(
            f"Corpus bulunamadı: {CORPUS_FILE}"
        )

    print(f"Corpus: {CORPUS_FILE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Minimum frequency: {MIN_FREQUENCY}")
    print()

    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]"
        )
    )

    tokenizer.normalizer = normalizers.NFC()

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
        ],
        show_progress=True,
    )

    print("Tokenizer eğitiliyor...")
    print("Bu işlem corpus boyutuna bağlı olarak uzun sürebilir.")
    print()

    tokenizer.train(
        files=[str(corpus)],
        trainer=trainer,
    )

    print()
    print("Tokenizer eğitimi tamamlandı.")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer_path = output_dir / "tokenizer.json"

    tokenizer.save(
        str(tokenizer_path)
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    fast_tokenizer.save_pretrained(
        str(output_dir)
    )

    print()
    print("Tokenizer bilgileri:")
    print(f"Vocab size : {fast_tokenizer.vocab_size}")
    print(f"PAD        : {fast_tokenizer.pad_token_id}")
    print(f"UNK        : {fast_tokenizer.unk_token_id}")
    print(f"BOS        : {fast_tokenizer.bos_token_id}")
    print(f"EOS        : {fast_tokenizer.eos_token_id}")

    test_texts = [
        "Merhaba dünya!",
        "Türkçe bir dil modeli geliştiriyorum.",
        "Çalışmalarımızı bugün tamamladık.",
        "Şimdi öğrenmeye başlıyoruz.",
        "Işık, ışıkta ve ışıltılı bir şekilde parlıyor.",
        "Ğ, ğ, Ü, ü, Ş, ş, İ, i, Ö, ö, Ç, ç, I, ı",
    ]

    print()
    print("Tokenizer testi:")

    for text in test_texts:
        encoded = fast_tokenizer(
            text,
            add_special_tokens=False
        )

        ids = encoded["input_ids"]

        tokens = fast_tokenizer.convert_ids_to_tokens(
            ids
        )

        decoded = fast_tokenizer.decode(
            ids,
            skip_special_tokens=True
        )

        print()
        print("Metin:")
        print(text)

        print("Tokenlar:")
        print(tokens)

        print("ID'ler:")
        print(ids)

        print("Decode:")
        print(decoded)

        print("Aynı:", text == decoded)

    print()
    print("Hugging Face Hub'a yükleniyor...")

    fast_tokenizer.push_to_hub(
        HF_REPO,
        private=False,
    )

    print()
    print(
        f"Tokenizer başarıyla yüklendi: "
        f"https://huggingface.co/{HF_REPO}"
    )


if __name__ == "__main__":
    main()

from datasets import load_dataset

ds = load_dataset(
    "tascib/turkish-llm-dataset",
    split="train",
    streaming=True
)

with open("corpus.txt", "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        f.write(row["text"] + "\n")

        if i + 1 >= 2_000_000:
            break

print("corpus.txt hazır!")

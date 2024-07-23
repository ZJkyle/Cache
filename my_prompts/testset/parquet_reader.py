import pandas as pd
import nltk

nltk.download("punkt")

accumulated_texts = []
total_tokens = 0
newline_cnt = 0
token_limit = 128
minimum_tokens_for_newline = 32
df = pd.read_parquet("original.parquet")


def add_newline_after_period(text):
    global newline_cnt
    tokens = nltk.word_tokenize(text)
    newline_cnt += len(tokens)
    if newline_cnt >= minimum_tokens_for_newline:
        text = text.replace(". ", ".\n", 1)
        newline_cnt = 0
    return text


while total_tokens < token_limit:
    sample_row = df.sample(n=1).iloc[0]
    title_with_newline = f"Title:\n{sample_row['title']}\n"
    formatted_text = add_newline_after_period(sample_row["text"])
    text_with_newline = f"Contents:\n{formatted_text}\n\n"
    full_text = f"{title_with_newline}{text_with_newline}"
    tokens = nltk.word_tokenize(full_text)
    accumulated_texts.append(full_text)
    total_tokens += len(tokens)

output_text = "".join(accumulated_texts)

file_name = f"{total_tokens}_tokens.txt"

with open(file_name, "w", encoding="utf-8") as file:
    file.write(output_text)

print(f"Total tokens: {total_tokens}")
print(f"Output saved to {file_name}")

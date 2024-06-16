import csv


def read_csv_column(file_path, column_name):
    """Read specified column from CSV file and return as a list of strings."""
    summaries = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            summaries.append(row[column_name])
    return summaries


def join_texts_with_newline(texts):
    """Join list of texts with newline character."""
    return "\n".join(texts)


def count_tokens(text):
    """Count the number of tokens in the text."""
    tokens = text.split()
    return len(tokens)


def write_to_file(text, output_file_path):
    """Write text to a file."""
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)


def process_csv(file_path, column_name, base_output_file_path, max_tokens=64000):
    summaries = read_csv_column(file_path, column_name)
    joined_text = ""
    total_token_count = 0
    file_index = 1

    for summary in summaries:
        current_token_count = count_tokens(summary)

        if total_token_count + current_token_count > max_tokens:
            output_file_path = f"{base_output_file_path}_{file_index}.txt"
            write_to_file(join_texts_with_newline([joined_text]), output_file_path)
            print(f"File {output_file_path} written with {total_token_count} tokens.")
            joined_text = ""
            total_token_count = 0
            file_index += 1

        joined_text += summary + "\n"
        total_token_count += current_token_count

    if joined_text:
        output_file_path = f"{base_output_file_path}_{file_index}.txt"
        write_to_file(join_texts_with_newline([joined_text]), output_file_path)
        print(f"File {output_file_path} written with {total_token_count} tokens.")


if __name__ == "__main__":
    csv_file_path = "input.csv"
    column_name = "summary"
    base_output_file_path = "output"

    process_csv(csv_file_path, column_name, base_output_file_path)

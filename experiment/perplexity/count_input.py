def count_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()
            word_count = len(words)
            return word_count
    except FileNotFoundError:
        return "File not found. Please check the path and try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage:
file_path = input("Enter the path to the .txt file: ")
word_count = count_words_in_file(file_path)
print(f"{word_count}")

import os
import lzma
import tarfile
import emoji
import re
from langdetect import detect
from tqdm import tqdm

# Define dataset paths
base_dir = "D:/Documents/HuffmanLLM/DATA/openwebtext/testtext"
output_dir = "D:/Documents/HuffmanLLM/DATA/openwebtext/testext_processed"
os.makedirs(output_dir, exist_ok=True)
FILES_PER_FOLDER = 10000
# Define thresholds
EMOJI_THRESHOLD = 0.05  # If >5% of characters are emojis, remove file


def is_mostly_english(text, min_words=10):
    """Returns True if the text is detected as mostly English."""
    words = text.split()
    if len(words) < min_words:
        return False  # Too short to determine language
    try:
        return detect(" ".join(words[:min_words])) == "en"
    except:
        return False  # If detection fails, assume non-English


def emoji_ratio(text):
    """Returns the percentage of characters that are emojis."""
    emoji_count = sum(1 for char in text if emoji.is_emoji(char))
    return emoji_count / max(len(text), 1)


def process_text(text):
    """Lowercases text and removes unwanted characters while preserving spaces."""
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F\s]+", " ", text)  # Keep ASCII characters and spaces
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces into a single space
    return text.strip()


def get_subfolder(count):
    """Returns the correct subfolder based on file count"""
    folder_index = count // FILES_PER_FOLDER
    subfolder = os.path.join(output_dir, f"{folder_index:04d}")  # e.g., "processed/0001/"
    os.makedirs(subfolder, exist_ok=True)
    return subfolder


# Process all .xz archives
def clean_corpus():
    num_saved = 0
    for xz_file in tqdm(os.listdir(base_dir), desc="Scanning .xz files"):
        xz_file_path = os.path.join(base_dir, xz_file)

        if xz_file.endswith(".xz"):
            with lzma.open(xz_file_path) as xz_stream:
                with tarfile.open(fileobj=xz_stream, mode="r:") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.endswith(".txt"):
                            file_name = os.path.basename(member.name)
                            subfolder = get_subfolder(num_saved)
                            output_file = os.path.join(subfolder, file_name)

                            f = tar.extractfile(member)
                            if f:
                                text = f.read().decode('utf-8', errors='ignore')

                                # Check language and emoji content
                                if not is_mostly_english(text) or emoji_ratio(text) > EMOJI_THRESHOLD:
                                    continue  # Skip non-English or emoji-heavy files

                                # Process text
                                cleaned_text = process_text(text)

                                # Save processed text
                                with open(output_file, "w", encoding="utf-8") as out_f:
                                    out_f.write(cleaned_text)

                                num_saved += 1

    print("✅ Preprocessing complete. Cleaned files saved in:", output_dir)


clean_corpus()

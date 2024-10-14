import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gc
import os
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from huggingface_hub.hf_api import HfFolder

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from datasets import Dataset

# Save Huggingface token
HfFolder.save_token("huggingface token")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# List of splits to process
splits = list(range(1, 101))
folder_name = "/data/sxs7285/Porjects_code/sector_activity/data/data_csv2"

# Check for existing summarized files and remove corresponding splits
splits_to_process = []
for split in splits:
    summarized_file_path = f'{folder_name}/linkedin_split_{str(split)}_summarized.csv'
    if not os.path.exists(summarized_file_path):
        print("ss", split)
        splits_to_process.append(split)

print(f"Splits to process: {splits_to_process}")

# Define summarization function using LexRankSummarizer
def summarize_text(text, sentences_count=6):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

# Multiprocessing function for summarization
def process_texts(texts):
    return [summarize_text(text, sentences_count=5) for text in texts]

# Function to save batch results
def save_batch_results(results, batch_index, folder_name):
    file_path = f"{folder_name}/batch_{batch_index}.csv"
    results.to_csv(file_path, index=False)

# Function to log system usage
def log_system_usage():
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2)} MB")
    print(f"CPU usage: {psutil.cpu_percent()}%")

# Loop through each split
for split in splits_to_process:
    start_time = time.time()  # Start timer
    print("Split1", split)
    json_file_path = f"/data/sxs7285/Porjects_code/sector_activity/data/data_json_new2/json_{str(split)}_final.json"

    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        print(f"File {json_file_path} does not exist. Skipping to the next split.")
        continue
    print("Split2", split)
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # Removing entries with 'sector_of_activity' as 'Not Found'
    filtered_data = {key: value for key, value in json_data.items()}
    print("Split3", split)

    # Extract relevant data directly from JSON
    contents = []
    labels = []
    coarse_labels = []
    urls = []

    for entry in filtered_data.values():
        contents.append(entry['content'])
        labels.append(entry['sector_of_activity'])
        coarse_labels.append(entry['sector_of_activity_coarse'])
        urls.append(entry['pp_url'])

    print("Split4", split)
    # Filter out rows with token_count > 100000
    valid_indices = [i for i, content in enumerate(contents) if len(content) <= 500000]
    print("Split5", split)

    valid_contents = [contents[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    valid_coarse_labels = [coarse_labels[i] for i in valid_indices]
    valid_urls = [urls[i] for i in valid_indices]
    print("Split6", split)

    # Set the number of processes to the number of available CPUs
    max_workers = 6  # Adjust the number of workers based on your system

    # Split texts into batches
    batch_size = len(valid_contents) // max_workers
    batches = [valid_contents[i * batch_size:(i + 1) * batch_size] for i in range(max_workers)]
    if len(valid_contents) % max_workers != 0:
        batches.append(valid_contents[max_workers * batch_size:])

    batch_index = 0
    print("Split7", split)

    # Use ProcessPoolExecutor to summarize the batches
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_texts, batches))

    batch_results = []
    for result in results:
        batch_results.extend(result)

    # Check for length mismatch
    if len(batch_results) != len(valid_contents):
        print(f"Length mismatch: batch_results = {len(batch_results)}, valid_contents = {len(valid_contents)}")

    # Prepare data for saving to CSV
    summary_data = {
        'content': valid_contents,
        'label': valid_labels,
        'label_coarse': valid_coarse_labels,
        'pp_url': valid_urls,
        'extractive_summary': batch_results
    }

    summary_df = pd.DataFrame(summary_data)

    # Save the result
    summary_df.to_csv(f'{folder_name}/linkedin_split_{str(split)}_summarized.csv', index=False)
 
    # Display the updated DataFrame
    print(summary_df)

    # Check for NaN values
    nan_count = summary_df['extractive_summary'].isna().sum()
    print(f"Number of NaN values in the column for split {split}:", nan_count)

    # Log system usage and force garbage collection
    log_system_usage()
    gc.collect()
    
    end_time = time.time()  # End timer
    total_time = end_time - start_time  # Calculate total time
    print(f"Total time for processing split {split}: {total_time} seconds")

import pandas as pd
import string
import matplotlib.pyplot as plt
import requests


def count_unique_words(file_url):
    response = requests.get(file_url)

    excluded_words = {'the', 'a', 'an', 'be'}
    text = response.text

    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    filtered_words = [word for word in words if word not in excluded_words]

    word_series = pd.Series(filtered_words)
    word_counts = word_series.value_counts()
    return word_counts

file_url = 'https://raw.githubusercontent.com/aalanwar/Logical-Zonotope/refs/heads/main/README.md'
word_count = count_unique_words(file_url)
print(word_count)

def plot_top_words(word_counts, top_n=10):
    # Get the top N words
    top_words = word_counts.head(top_n)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    top_words.plot(kind='bar', color='purple')
    plt.title('Top 10 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()

plot_top_words(word_count)

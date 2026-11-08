# Hansard Word Usage & Sentiment Analysis

This project analyzes and compares the sentiment and politeness/rudeness of historical versus modern political debates using custom-trained `Word2Vec` embeddings and an adapted `VADER` sentiment analysis lexicon.

## Project Structure

```text
├── Collection of Data/
│   ├── old debates/        # XML files for historical debates
│   └── new debates/        # XML files for modern debates
├── 1800-vocab.pkl          # Vocabulary pickle file for historical embeddings
├── 1800-w.npy              # NumPy embedding matrix for historical words
└── Word Usage Analyser.py  # Main Python analysis script

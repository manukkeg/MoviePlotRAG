# MoviePlotRAG

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions about movie plots using the Wikipedia Movie Plots dataset.

## Requirements

- Python 3.8+
- HuggingFace API token

## Installation

1. Clone the repository:
Clone GitHub repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download `wiki_movie_plots_deduped.csv` Dataset
   - Place it in the project root directory

4. Create a `.env` file in the root directory:
   - HF_TOKEN=your_huggingface_token_here

## Usage
Run the program
```bash
python main.py
```
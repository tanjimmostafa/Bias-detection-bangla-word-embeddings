# Detecting Gender Bias in Word Embeddings

This repository contains code for detecting gender bias in word embeddings, specifically focusing on a Bangla Word2Vec model. It explores different methods to identify and quantify bias present in the word representations.

## Project Structure

The code is presented as a Google Colab notebook, demonstrating the step-by-step process of loading the model, exploring its vocabulary, and performing bias detection.

## Methods Used

The notebook employs the following techniques to detect gender bias:

1.  **Checking Vocabulary:** Verifying the presence of specific gender-related words in the Word2Vec model's vocabulary.
2.  **Gender Direction Projection:** Identifying words that are most similar to a predefined "gender direction" (calculated as the difference between male and female word vectors). This helps to see which words are strongly associated with one gender over the other.
3.  **Principal Component Analysis (PCA):** Applying PCA to the word embeddings of gender-specific terms to identify the principal components that capture the most variance. The weights of these components can reveal gender-related dimensions in the embedding space. This is performed for both English and Bangla words.
4.  **Cosine Similarity in the Gender Subspace:** Calculating the cosine similarity between gender-specific words to understand their proximity in the embedding space.
5.  **Permutation Test:** A statistical test to determine if the observed cosine similarity between gender-specific words is significantly different from what would be expected by chance.

## Getting Started

To run the code, you can-

1.  Open the provided code in Google Colab.
2.  Ensure you have the necessary libraries installed (bnlp\_toolkit, gensim, numpy, scikit-learn, matplotlib). The notebook includes commands to install `bnlp_toolkit`.
3.  Download the Bangla Word2Vec model. The notebook provides the Wget command to download the model from Hugging Face.
4.  If running the PCA for Bangla words and encountering font issues, download a Bangla font like Kalpurush and mount your Google Drive to access it, as demonstrated in the notebook.
5.  Execute the cells sequentially.

## Code Description

-   **Loading Bangla Word2vec Model:** Installs `bnlp_toolkit`, creates a directory, downloads and unzips the Word2Vec model.
-   **Checking the words in the Word2vec Vocab:** Loads the model and checks if a predefined list of words exists in the vocabulary.
-   **Detecting Bias by Finding Most Common Word via Gender Direction Projection:** Calculates the gender direction vector and computes the similarity of gender-specific words to this direction.
-   **Detecting Bias Performing PCA:** Performs PCA on English gender-specific words and visualizes the weights.
-   **Detecting Bias For Bangla Word using PCA:** Mounts Google Drive to access a Bangla font, performs PCA on Bangla gender-specific words, and visualizes the weights using the specified font.
-   **Detecting Bias For Bangla Word using Cosine Similarity in the Gender Subspace:** Calculates and prints the cosine similarity matrix for Bangla gender-specific words.
-   **Permutation Test:** Defines and calls a function to perform a permutation test on the cosine similarity of the Bangla gender-specific words.

## Dependencies

-   `bnlp_toolkit`
-   `gensim`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`

## Word2Vec Model

The Bangla Word2Vec model used in this project is sourced from Hugging Face: `sagorsarker/bangla_word2vec`.

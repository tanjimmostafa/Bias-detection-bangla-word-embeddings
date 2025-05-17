# Detecting Bias in Bangla Word Embeddings

This repository contains code and experiments for detecting bias—particularly gender and religious bias—in Bangla word embeddings. The project utilizes both classical word embedding models (Word2Vec, GloVe) and modern transformer-based models (Bangla BERT) to analyze bias using a variety of statistical, visualization, and machine learning techniques.

## Project Structure

All experiments and analyses are provided as a Google Colab notebook (`Detecting_Bias.ipynb`). The notebook demonstrates the step-by-step process of:

- Downloading and loading Bangla Word2Vec and GloVe models.
- Exploring vocabularies.
- Performing bias detection using multiple methods.
- Applying transformer-based models (Bangla BERT) for advanced bias and context analysis.

## Methods Used

The notebook employs the following techniques to detect and analyze bias:

1. **Checking Vocabulary:**  
   Verifies the presence of specific gender- and religion-related words in the vocabularies of the Word2Vec and GloVe models.

2. **Gender/Religion Direction Projection:**  
   Identifies words most similar to a predefined "direction" (e.g., the vector difference between 'ইসলাম' and 'খ্রিস্টান' or between male and female terms). This technique surfaces words more associated with specific social groups.

3. **Principal Component Analysis (PCA):**  
   Applies PCA to the embeddings of gender- or religion-specific words (in both English and Bangla). Visualizes the principal components and their weights, often using Bangla fonts for clarity.

4. **Cosine Similarity in Gender/Religion Subspace:**  
   Calculates cosine similarity matrices between gender- or religion-specific words to understand their relationships in the embedding space.

5. **Permutation Test with Visualization:**  
   Performs a statistical permutation test to determine if observed similarities (e.g., between gender-specific words) are significant. Visualizes the null distribution and observed values using histograms.

6. **Word Embedding Association Test (WEAT):**  
   Implements several variants of the WEAT metric for both Word2Vec and GloVe. WEAT quantifies the association between sets of target and attribute words, with examples for both Bangla and English. Improved and alternative implementations are included.

7. **Masked Word Prediction (Word2Vec & Transformers):**  
   - **Word2Vec:** Predicts the most probable word for a masked position in a sentence based on context.
   - **Bangla BERT:** Uses HuggingFace Transformers to predict masked tokens in context, leveraging the power of transformer-based language models.

8. **Transformer-Based Analysis (Bangla BERT):**  
   Loads and utilizes Bangla BERT for masked language modeling, context-driven word prediction, and bias analysis.

## Getting Started

To run the code:

1. **Open the notebook in Google Colab.**
2. **Install dependencies:**  
   The notebook includes commands to install all necessary Python packages (see the Dependencies section).
3. **Download pretrained models:**  
   The notebook provides commands and links to download Bangla Word2Vec and GloVe models from Hugging Face.
4. **(Optional) Download Bangla font:**  
   For PCA visualizations, download a Bangla font (e.g., Kalpurush) and mount your Google Drive if required.
5. **Run the notebook cells sequentially** to follow the full bias detection and analysis pipeline.

## Code Description

- **Loading Bangla Word2Vec and GloVe Models:**  
  Installs `bnlp_toolkit`, creates directories, downloads, and unzips the models for use.
- **Vocabulary Checking:**  
  Loads models and checks for the presence of key gender/religion-related words.
- **Bias Detection Methods:**  
  - **Direction Projection:** Finds most associated words with gender or religion vectors.
  - **PCA & Visualization:** Applies PCA to gender/religion word sets and visualizes results.
  - **Cosine Similarity:** Computes similarity matrices between word groups.
  - **Permutation Test:** Performs and visualizes the statistical significance of observed bias.
  - **WEAT Scoring:** Multiple implementations for both Word2Vec and GloVe, with support for Bangla and English word sets.
  - **Masked Word Prediction:** Predicts words for masked positions using both Word2Vec and Bangla BERT.
  - **Transformer Analysis:** Loads Bangla BERT for advanced bias and context analysis.

## Example: WEAT and Masked Word Prediction

- **WEAT Example:**
  ```python
  # Example word sets (Bangla or English)
  target_X = ['doctor', 'engineer', 'scientist']
  target_Y = ['nurse', 'teacher', 'librarian']
  attribute_A = ['intelligent', 'logical', 'analytical']
  attribute_B = ['caring', 'empathetic', 'nurturing']

  # Calculate WEAT score using the provided function in the notebook
  score = weat_score(word2vec_model, target_X, target_Y, attribute_A, attribute_B)
  print(f'WEAT Score: {score}')
  ```

- **Masked Word Prediction Example (Word2Vec):**
  ```python
  sentence = ['ছেলেটি', 'খেলতে', '[MASK]', 'মাঠে', 'গেল']
  masked_index = 2
  predicted_word = predict_masked_word(sentence, masked_index)
  print(f"Predicted word: {predicted_word}")
  ```

- **Masked Token Prediction Example (Bangla BERT):**
  ```python
  sentence = "[MASK] মাঠে গেল।"
  predicted_tokens = predict_masked_token(sentence)
  print("Predictions:", predicted_tokens)
  ```

## Dependencies

- `bnlp_toolkit`
- `gensim`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `transformers`
- `torch`

## Pretrained Models

The Bangla embedding models used in this project are sourced from Hugging Face:

- [sagorsarker/bangla_word2vec](https://huggingface.co/sagorsarker/bangla_word2vec)
- [sagorsarker/bangla-glove-vectors](https://huggingface.co/sagorsarker/bangla-glove-vectors)
- [csebuetnlp/banglabert](https://huggingface.co/csebuetnlp/banglabert)
- [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base)

## Citation

If you use this repository or notebook in your research, please cite appropriately.

---

For detailed explanations, code, and outputs, refer to the [`Detecting_Bias.ipynb`](./Detecting_Bias.ipynb) notebook in this repository.

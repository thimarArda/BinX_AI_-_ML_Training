# NLP Preprocessing

This notebook demonstrates a comprehensive NLP preprocessing pipeline for text data, focusing on the Banking77 dataset. It covers essential techniques such as tokenization, lowercasing, punctuation removal, stop-word handling, and lemmatization, emphasizing the importance of preserving critical information like negations.

## Notebook Contents

| Section | Title                                        | Content                                                                                                                   |
| :------ | :------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| 1       | Why Does Text Need Preprocessing?            | Explains why raw text is challenging for ML models and introduces common preprocessing techniques.                        |
| 2       | Understanding the Banking Dialogue Dataset   | Introduces the Banking77 dataset, its structure, and inspects its initial examples and columns.                           |
| 3       | Selecting Raw Text for Preprocessing         | Demonstrates selecting a few sample texts and a specific sentence for manual preprocessing experimentation.               |
| 4       | Text Processing Techniques                   | Details various text preprocessing methods, including tokenization, lowercasing, punctuation removal, stop-word handling, and lemmatization. |
| 5       | Applying the Full Cleaning Pipeline          | Combines all individual preprocessing steps into a single function and tests it on a sample text.                         |
| 6       | Applying the Pipeline to Banking77           | Applies the preprocessing pipeline to the entire Banking77 training dataset and checks if important information, like negations, is preserved. |
| 7       | Day 1 Summary                                | Summarizes the work done on Day 1 of Sprint 3, covering NLP preprocessing techniques and their application.               |

## How to Use this Notebook in Google Colab

1.  **Open in Colab**: Click the "Open in Colab" badge (if available) or upload the `.ipynb` file directly to your Google Drive and open it with Colaboratory.
2.  **Run Cells Sequentially**: Execute each code cell in order from top to bottom. You can do this by selecting a cell and pressing `Shift + Enter`, or by using the "Runtime" menu to "Run all" or "Run selected cells".
3.  **Install Dependencies**: Ensure you run the cell containing `!pip install -U transformers datasets==2.18.0 accelerate evaluate scikit-learn` at the beginning to install all necessary libraries.

### Guidelines to Install the Dataset

Due to recent changes in the `datasets` library, directly loading the `PolyAI/banking77` dataset using `load_dataset("PolyAI/banking77")` might encounter errors related to unsupported dataset scripts. This notebook handles dataset loading by:

1.  **Direct Download**: The notebook directly downloads the `train.json` and `test.json` files for the Banking77 dataset from the PolyAI GitHub repository using the `requests` library.
2.  **DataFrame Conversion**: The downloaded JSON data is then converted into Pandas DataFrames.
3.  **DatasetDict Creation**: Finally, these DataFrames are used to create a `DatasetDict` object, mimicking the structure expected by the `datasets` library for seamless integration with subsequent NLP operations.


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filepath):
    # Load data from a CSV file.
    try:
        return pd.read_csv(filepath, encoding='ISO-8859-1')
    except FileNotFoundError:
        raise Exception(f"File not found at {filepath}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")


def compute_tfidf_matrix(data):
    # Convert the text data to a TF-IDF matrix.
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(data['Sentence'])


def compute_similarity(matrix):
    # Calculate the cosine similarity matrix from the TF-IDF matrix.
    return cosine_similarity(matrix)


def filter_similar_items(similarity_matrix, threshold=0.3):
    # Identify items with similarity above the threshold to filter out.
    rows_to_delete = set()
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                rows_to_delete.add(j)
    return rows_to_delete


def save_filtered_data(data, rows_to_delete, filepath):
    # Save the filtered data to a CSV file.
    df_filtered = data.drop(rows_to_delete)
    df_filtered.to_csv(filepath, index=False)


def main():
    df = load_data('INPUT_CSV_PATH')
    tfidf_matrix = compute_tfidf_matrix(df)
    similarity_matrix = compute_similarity(tfidf_matrix)
    rows_to_delete = filter_similar_items(similarity_matrix)
    save_filtered_data(df, rows_to_delete, 'OUTPUT_CSV_PATH')


if __name__ == "__main__":
    main()

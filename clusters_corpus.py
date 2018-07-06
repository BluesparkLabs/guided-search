from math import log10
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.downloader import download
from nltk.stem.porter import PorterStemmer
from pymarc import MARCReader
from scipy.sparse.csr import csr_matrix
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sys import exit
import hashlib
import numpy
import os
import pandas
import sqlite3
import string

numpy.set_printoptions(threshold=numpy.nan)

def main():
    documents = indexed_documents()
    total_docs = len(documents)
    # We generate one cluster for each 500 docs.
    num_clusters = round(total_docs / 500)

    # Load vectorize from dump or process documents vectorization
    try:
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError:
        matrix, vectorizer = documents_vectors()
        joblib.dump(vectorizer, 'vectorizer.pkl')

    # Load cluster model from dump or process clustering.
    try:
        km = joblib.load('doc_cluster.pkl')
    except FileNotFoundError:
        km = KMeans(n_clusters=num_clusters)
        km.fit(matrix)
        joblib.dump(km, 'doc_cluster.pkl')

    terms = vectorizer.get_feature_names()
    clusters = km.labels_.tolist()
    centroids = km.cluster_centers_.argsort()[:, ::-1]
    frame = pandas.DataFrame(documents, index = [clusters] , columns = ['doc_id'])

    for i in range(num_clusters):
        print("Cluster %d:" % (i))
        for word_idx in centroids[i, 0:9]:
            word = terms[word_idx]
            print(' %s' % (word), end=',')
        print("\n")

        print("Documents:")
        for doc_id in frame.ix[i]['doc_id'].values.tolist():
            print(' - %s' % (document_field_value(doc_id, 'body')))

    print("\n")
    print("====================================")

def db_connect():
    """ Connect to DB and init tables schema. """
    # Documents table schema.
    create_documents_table = '''CREATE TABLE IF NOT EXISTS documents
    (id VARCHAR(40) PRIMARY KEY, body LONGTEXT)'''
    # Documents words index schema.
    create_index_table = '''CREATE TABLE IF NOT EXISTS documents_words
    (id VARCHAR(40), word VARCHAR)'''
    # Document ID index to speed up word retrievals.
    create_index_table_index = '''CREATE INDEX IF NOT EXISTS
    document_id ON documents_words (id)'''
    # Connect to DB and create the tables.
    connection = sqlite3.connect('./index.sqlite')
    db = connection.cursor()
    db.execute(create_documents_table)
    db.execute(create_index_table)
    db.execute(create_index_table_index)
    return connection

def documents_vectors():
    """ Builds indexed documents words vectors.

    :returns: List with corpus words matrix and the vectorizer object.
    :rtype: csr_matrix, TfidfVectorizer

    """

    documents = indexed_documents()
    total_documents = len(documents)
    print("Processing %d documents." % (total_documents))
    # Filter terms that appears in more than 99% of the documents
    # and terms that do not appear on at least 1% of the documents.
    vectorizer = TfidfVectorizer(tokenizer=indexed_document_words,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.99,
                                 min_df=0.01,
                                 lowercase=True)
    matrix = vectorizer.fit_transform(documents) # type: csr_matrix
    return matrix, vectorizer

def indexed_documents():
    """ Get indexed documents list.

    :returns: A list of indexed documents.

    """

    db = connection.cursor()
    db.execute('''SELECT id FROM documents''')
    result = db.fetchall()

    return [row[0] for row in result]

def document_field_value(doc_id, field_name):
    """ Get indexed document field value.

    :param str doc_id: The document ID.
    :param str field_name: The document field name to get value from.
    :returns: The field value.
    :rtype: str

    """
    db = connection.cursor()
    db.execute('''SELECT {} FROM documents WHERE id = ?'''.format(field_name), (doc_id,))
    result = db.fetchone()
    return result[0]

def indexed_document_words(doc_id):
    """ Get indexed document words.

    :param str doc_id: The document ID.
    :returns: A list of document words.

    """

    print("Tokens for document '%s'" % (doc_id))
    # Get document words
    db = connection.cursor()
    db.execute('''SELECT word FROM documents_words WHERE id = ?''', (doc_id,))
    result = db.fetchall()
    # Extract the first column from all rows.
    document_words = [row[0] for row in result]
    return document_words

def corpus_words():
    """Extract document words from DB index.

    :returns: A list of all unique documents words.

    """

    all_words = []
    db = connection.cursor()
    # Count total documents.
    db.execute('''SELECT COUNT(id) as total FROM documents''')
    total_docs = db.fetchone()[0]
    # Get words list with frequency filtering words that only occurs in 2 or
    # less documents that are not relevant.
    db.execute('''SELECT word, count(word) AS frequency FROM documents_words
               GROUP BY WORD HAVING frequency >= 5''')
    result = db.fetchall()

    for row in result:
        # Calculate inverse document frequency.
        idf = round(log10((total_docs + 1) / (row[1] + 1)), 2)
        # Remove terms that are not very relevant to reduce the words
        # vector size, based on manual inspection below idf 2.5 seems
        # that words are too frequent to be considered representative.
        # See: https://moz.com/blog/inverse-document-frequency-and-the-importance-of-uniqueness
        if idf <= 3:
            continue
        # Keep the work for vectorization.
        all_words.append(row[0])

    return all_words

if __name__ == "__main__":
    connection = db_connect()
    main()

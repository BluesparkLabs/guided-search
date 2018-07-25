from cleo import Command
from math import log10
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sys import exit
import numpy
import os
import pandas
import sqlite3

numpy.set_printoptions(threshold=numpy.nan)


class ClusterCorpusCommand(Command):
    """
    Clustering of corpus documents.

    guided-search:cluster

    """

    def handle(self):
        """
        Process clustering of corpus documents.

        """

        self.connection = self.db_connect()
        documents = self.indexed_documents()
        total_docs = len(documents)
        # We generate one cluster for each 500 docs.
        num_clusters = round(total_docs / 500)

        # Load vectorize from dump or process documents vectorization
        try:
            vectorizer = joblib.load('vectorizer.pkl')
        except FileNotFoundError:
            matrix, vectorizer = self.documents_vectors()
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
        frame = pandas.DataFrame(
            documents,
            index=[clusters],
            columns=['doc_id']
        )

        for i in range(num_clusters):
            print("Cluster %d:" % (i))
            for word_idx in centroids[i, 0:9]:
                word = terms[word_idx]
                print(' %s' % (word), end=',')
            print("\n")

            print("Documents:")
            for doc_id in frame.ix[i]['doc_id'].values.tolist():
                print(' - %s' % (self.document_field_value(doc_id, 'body')))

        print("\n")
        print("====================================")

    def db_connect(self):
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

    def documents_vectors(self):
        """ Builds indexed documents words vectors.

        :returns: List with corpus words matrix and the vectorizer object.
        :rtype: csr_matrix, TfidfVectorizer

        """

        documents = self.indexed_documents()
        total_documents = len(documents)
        print("Processing %d documents." % (total_documents))
        # Filter terms that appears in more than 99% of the documents
        # and terms that do not appear on at least 1% of the documents.
        vectorizer = TfidfVectorizer(
            tokenizer=indexed_document_words,
            stop_words=stopwords.words('english'),
            max_df=0.99,
            min_df=0.01,
            lowercase=True
        )
        matrix = vectorizer.fit_transform(documents)  # type: csr_matrix
        return matrix, vectorizer

    def indexed_documents(self):
        """ Get indexed documents list.

        :returns: A list of indexed documents.

        """

        db = self.connection.cursor()
        db.execute('''SELECT id FROM documents''')
        result = db.fetchall()

        return [row[0] for row in result]

    def document_field_value(self, doc_id, field_name):
        """ Get indexed document field value.

        :param str doc_id: The document ID.
        :param str field_name: The document field name to get value from.
        :returns: The field value.
        :rtype: str

        """
        db = self.connection.cursor()
        db.execute(
            '''SELECT {} FROM documents WHERE id = ?'''.format(field_name),
            (doc_id,)
        )
        result = db.fetchone()
        return result[0]

    def corpus_words(self):
        """Extract document words from DB index.

        :returns: A list of all unique documents words.

        """

        all_words = []
        db = self.connection.cursor()
        # Count total documents.
        db.execute('''SELECT COUNT(id) as total FROM documents''')
        total_docs = db.fetchone()[0]
        # Get words list with frequency filtering words that only occurs in 2
        # or less documents that are not relevant.
        db.execute('''SELECT word, count(word) AS frequency FROM documents_words
                GROUP BY WORD HAVING frequency >= 5''')
        result = db.fetchall()

        for row in result:
            # Calculate inverse document frequency.
            idf = round(log10((total_docs + 1) / (row[1] + 1)), 2)
            # Remove terms that are not very relevant to reduce the words
            # vector size, based on manual inspection below idf 2.5 seems that
            # words are too frequent to be considered representative.  See:
            # https://moz.com/blog/inverse-document-frequency-and-the-importance-of-uniqueness
            if idf <= 3:
                continue
            # Keep the work for vectorization.
            all_words.append(row[0])

        return all_words

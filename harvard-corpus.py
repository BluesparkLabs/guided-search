from math import log10
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from pymarc import MARCReader
from sys import exit
import hashlib
import nltk.corpus
import numpy
import os
import sqlite3
import string

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
numpy.set_printoptions(threshold=numpy.nan)

# Process the Harvard Dataset and load as NLTK corpus.
def main():

    data_dir = '/Users/pablocc/harvard_data/'
    counter = 0
    connection = db_connect()
    documents_vectors = documents_vectors(connection)
    print(documents_vectors)
    exit()

    for filename in os.listdir(data_dir):
        if os.path.isdir(data_dir + filename) or filename[0] == '.':
            continue

        with open(data_dir + filename, 'rb') as fh:
            reader = MARCReader(fh)
            for record in reader:
                document = prepare_record(record)
                counter += 1
                print("%s - processing document %s."
                      % (counter, document['id']))
                index_document(connection, document)


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

def index_document(connection, document):
    """ Store document words on DB. """

    db = connection.cursor()

    try:
        # Check if document exists.
        db.execute("SELECT id FROM documents WHERE id = ?", (document['id'],))
        result = db.fetchone()

        # Skip indexation if document exists.
        if result:
            print("Document %s is already indexed." % (document['id']))
            return

        # Extract document words.
        words_extract(document)

        db.execute('INSERT INTO documents (id, body) VALUES (?, ?)',
                   (document['id'], document['body']))

        for word in document['words']: # type: str
            # Skip short words.
            if len(word) <= 2:
                continue
            print("%s - %s" % (document['id'], word))
            db.execute('INSERT INTO documents_words (id, word) VALUES (?, ?)',
                       (document['id'], word))

        # Commit inserts.
        connection.commit()
    except sqlite3.Error as err:
        print("Error occurred: %s" % err)
        exit()

def words_extract(document):
    stemmer = PorterStemmer()
    body = document['body']
    # Remove punctuation.
    translator = str.maketrans('', '', string.punctuation)
    body = body.translate(translator)
    # Tokenize document words.
    words = word_tokenize(body)
    # Words stemming.
    words_root = [stemmer.stem(word) for word in words]
    # Save document words for vectorization phase.
    document['words'] = words_root

def documents_vectors(connection):
    """ Builds indexed documents words vectors. """

    vectors = []
    all_words = corpus_words(connection)
    documents = indexed_documents(connection)
    db = connection.cursor()

    for doc_id in documents:
        # Get document words
        db.execute('''SELECT word FROM documents_words WHERE id = ?''', doc_id)
        result = db.fetchall()
        # Extract the first column from all rows.
        doc_words = [row[0] for row in result]
        # Create document words vector.
        vectors[doc_id] = numpy.array(
            [word in doc_words for word in all_words],
            numpy.short
        )

    return vectors

def prepare_record(record):
    pubplace = clean(record['260']['a']) if '260' in record else None
    extent = clean(record['300']['a'], True) if '300' in record else None
    dimensions = record['300']['c'] if '300' in record else None
    subject = record['650']['a'] if '650' in record else None
    inclusiondate = record['988']['a'] if '988' in record else None
    source = record['906']['a'] if '906' in record else None
    library = record['690']['5'] if '690' in record else None
    notes = " ".join([field['a'] for field in record.notes() if 'a' in field])

    # Store fields on document array.
    document_fields = [
            record.isbn(),
            get_title(record),
            clean(record.author(), True),
            clean(record.publisher()),
            pubplace,
            clean(record.pubyear()),
            extent,
            dimensions,
            subject,
            inclusiondate,
            source,
            library,
            notes]

    # Concatenate all fields into string.
    body = ' '.join(list(filter(None.__ne__, document_fields)))
    print(body)
    docid = hashlib.md5(body.encode('utf-8')).hexdigest()
    document = {'id': docid, 'body': body}
    return document

# Get record title.
def get_title(record):
    if '245' in record and 'a' in record['245']:
        title = clean(record['245']['a'])
        if 'b' in record['245']:
            title += ' ' + clean(record['245']['b'])
        return title
    else:
        return None

# Clean unwanted characters on a field.
def clean(element, isAuthor=False):
    if element is None or not element.strip():
        return None
    else:
        element = element.strip()

        for character in [',', ';', ':', '/']:
            if element[-1] == character:
                return element[:-1].strip()

        if not isAuthor and element[-1] == '.':
            return element[:-1].strip()

        return element.strip()

def indexed_documents(connection):
    """ Get indexed documents list.

    :param sqlite3.Connection connection: DB connection.
    :returns: A list of indexed documents.

    """

    db = connection.cursor()
    db.execute('''SELECT id FROM documents''')
    result = db.fetchall()

    return result

def corpus_words(connection):
    """Extract document words from DB index.

    :param sqlite3.Connection connection: DB connection.
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
    main()

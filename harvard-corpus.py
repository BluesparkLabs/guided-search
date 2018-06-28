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
corpus_words = set()
documents = {}

# Process the Harvard Dataset and load as NLTK corpus.
def main():

    data_dir = '/Users/pablocc/harvard_data/'
    counter = 0
    connection = db_connect()

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

        for word in document['words']:
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

def document_vector(document):
    """ Builds a document words vector. """
    vector = numpy.array([
        word in words_root and not word in stop_words
        for word in words], numpy.short)
    return words_root

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

if __name__ == "__main__":
    main()

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from pymarc import MARCReader
from sys import exit
import hashlib
import nltk.corpus
import numpy
import os
import sqlite3

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
corpus_words = set()
documents = {}

# Process the Harvard Dataset and load as NLTK corpus.
def main():

    data_dir = '/Users/pablocc/harvard_data/'
    counter = 0
    db = db_connect()

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
                index_document(db, document)

def db_connect():
    """ Connect to DB and init tables schema. """
    # Documents table schema.
    create_documents_table = '''CREATE TABLE IF NOT EXISTS documents
    (id VARCHAR(40) PRIMARY KEY, document LONGTEXT)'''
    # Documents words index schema.
    create_index_table = '''CREATE TABLE IF NOT EXISTS documents_words
    (id VARCHAR(40) PRIMARY KEY, word VARCHAR)'''
    # Connect to DB and create the tables.
    conn = sqlite3.connect('./index.sqlite')
    db = conn.cursor()
    db.execute(create_documents_table)
    db.execute(create_index_table)
    return db

def index_document(db, document):
    """ Store document words on DB. """

    # Extract document words.
    words_extract(document)

    try:
        db.execute('INSERT INTO documents_words (id, body) VALUES (?, ?)',
                   (document['id'], document['body']))
    except sqlite3.Error as err:
        print("Error occurred: %s" % err)
    else:
        print(db.rowcount)

def words_extract(document):
    stemmer = PorterStemmer()
    # Tokenize document words.
    words = word_tokenize(document['body'])
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

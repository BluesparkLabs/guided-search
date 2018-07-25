from cleo import Command
from indexdb import IndexDB
from nltk import word_tokenize
from nltk.downloader import download
from nltk.stem.porter import PorterStemmer
from pymarc import MARCReader
from sys import exit
import hashlib
import os
import string

class IndexCorpusCommand(Command):
    """
    Index corpus documents.

    guided-search:index

    """

    def handle(self):
        """
        Process corpus documents indexation.

        """

        download('stopwords')
        indexdb = IndexDB()
        self.connection = indexdb.handler()
        data_dir = '/Users/pablocc/harvard_data/'
        counter = 0

        for filename in os.listdir(data_dir):
            if os.path.isdir(data_dir + filename) or filename[0] == '.':
                continue

            with open(data_dir + filename, 'rb') as fh:
                reader = MARCReader(fh)
                for record in reader:
                    document = self.prepare_record(record)
                    counter += 1
                    print("%s - processing document %s."
                        % (counter, document['id']))
                    self.index_document(document)

    def prepare_record(self, record):
        pubplace = self.clean(record['260']['a']) if '260' in record else None
        extent = self.clean(record['300']['a'], True) if '300' in record else None
        dimensions = record['300']['c'] if '300' in record else None
        subject = record['650']['a'] if '650' in record else None
        inclusiondate = record['988']['a'] if '988' in record else None
        source = record['906']['a'] if '906' in record else None
        library = record['690']['5'] if '690' in record else None
        notes = " ".join([field['a'] for field in record.notes() if 'a' in field])

        # Store fields on document array.
        document_fields = [
                record.isbn(),
                self.get_title(record),
                self.clean(record.author(), True),
                self.clean(record.publisher()),
                pubplace,
                self.clean(record.pubyear()),
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
    def get_title(self, record):
        if '245' in record and 'a' in record['245']:
            title = self.clean(record['245']['a'])
            if 'b' in record['245']:
                title += ' ' + self.clean(record['245']['b'])
            return title
        else:
            return None

    # Clean unwanted characters on a field.
    def clean(self, element, isAuthor=False):
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

    def words_extract(self, document):
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

    def index_document(self, document):
        """ Store document words on DB. """

        db = self.connection.cursor()

        try:
            # Check if document exists.
            db.execute("SELECT id FROM documents WHERE id = ?", (document['id'],))
            result = db.fetchone()

            # Skip indexation if document exists.
            if result:
                print("Document %s is already indexed." % (document['id']))
                return

            # Extract document words.
            self.words_extract(document)

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
            self.connection.commit()
        except sqlite3.Error as err:
            print("Error occurred: %s" % err)
            exit()

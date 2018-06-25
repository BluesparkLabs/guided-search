from pymarc import MARCReader
import numpy
import os
from sys import exit
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


# Process the Harvard Dataset and load as NLTK corpus.
def main():

    stemmer = PorterStemmer()
    data_dir = '/Users/pablocc/harvard_data/'

    for filename in os.listdir(data_dir):
        if os.path.isdir(data_dir + filename) or filename[0] == '.':
            continue

        with open(data_dir + filename, 'rb') as fh:
            reader = MARCReader(fh)
            for record in reader:
                document = prepare_record(record)
                words = word_tokenize(document)
                # Words stemming.
                words_root = [stemmer.stem(word) for word in words]
                print(words_root)

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
    document = ' '.join(list(filter(None.__ne__, document_fields)))
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

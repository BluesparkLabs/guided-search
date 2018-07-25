#!/usr/bin/env python

from index_corpus import IndexCorpusCommand
from cluster_corpus import ClusterCorpusCommand
from cleo import Application

application = Application()
application.add(IndexCorpusCommand())
application.add(ClusterCorpusCommand())

def indexed_document_words(self, doc_id):
    """ Get indexed document words.

    :param str doc_id: The document ID.
    :returns: A list of document words.

    """

    print("Tokens for document '%s'" % (doc_id))
    # Get document words
    db = self.connection.cursor()
    db.execute(
        '''SELECT word FROM documents_words WHERE id = ?''',
        (doc_id,)
    )
    result = db.fetchall()
    # Extract the first column from all rows.
    document_words = [row[0] for row in result]
    return document_words

if __name__ == '__main__':
    application.run()

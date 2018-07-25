import sqlite3


class IndexDB(object):

    """ Initialize index DB and connection. """

    def __init__(self):
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
        self._connection = connection
        db = self.connection.cursor()
        db.execute(create_documents_table)
        db.execute(create_index_table)
        db.execute(create_index_table_index)

    @property
    def connection(self):
        """ Return connection handler.  """
        return self._connection

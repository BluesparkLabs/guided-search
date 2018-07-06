#!/usr/bin/env python

from index_corpus import IndexCorpusCommand
from cleo import Application

application = Application()
application.add(IndexCorpusCommand())

if __name__ == '__main__':
    application.run()

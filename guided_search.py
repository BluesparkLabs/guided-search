#!/usr/bin/env python

from cleo import Application
from cluster_corpus import ClusterCorpusCommand
from index_corpus import IndexCorpusCommand
from indexdb import IndexDB
from os import sys

if __name__ == '__main__':
    application = Application()
    application.add(IndexCorpusCommand())
    application.add(ClusterCorpusCommand())
    application.run()

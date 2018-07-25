#!/usr/bin/env python

from index_corpus import IndexCorpusCommand
from cluster_corpus import ClusterCorpusCommand
from cleo import Application

if __name__ == '__main__':
    application = Application()
    application.add(IndexCorpusCommand())
    application.add(ClusterCorpusCommand())
    application.run()

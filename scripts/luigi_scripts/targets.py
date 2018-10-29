import luigi
import os
import zarr
import json
import pymongo
import logging

logger = logging.getLogger(__name__)

class FileTarget(luigi.Target):

    def __init__(self, filename):
        self.filename = filename

    def exists(self):
        isfile = os.path.isfile(self.filename)
        return isfile

class N5DatasetTarget(luigi.Target):

    def __init__(self, filename, dataset):
        self.filename = filename
        self.dataset = dataset

    def exists(self):

        logger.debug("Checking if %s exists...", self.filename)
        if not os.path.isdir(self.filename):
            logger.debug("%s does NOT exist", self.filename)
            return False

        logger.debug(
            "Checking if dataset %s is in %s...",
            self.dataset,
            self.filename)
        try:
            f = zarr.open(self.filename, mode='r')
            exists = self.dataset in f
        except:
            return False

        if not exists:
            logger.debug("%s is NOT contained in %s", self.dataset, self.filename)
        else:
            logger.debug("%s is contained in %s", self.dataset, self.filename)
        return exists

class N5AttributeTarget(luigi.Target):

    def __init__(self, filename, dataset, attribute):
        self.filename = filename
        self.dataset = dataset
        self.attribute = attribute

    def exists(self):
        if not os.path.isdir(self.filename):
            return False
        try:
            with z5py.File(self.filename, 'r') as f:
                return (self.dataset in f and self.attribute in f[self.dataset].attrs)
        except:
            return False

class JsonTarget(luigi.Target):

    def __init__(self, filename, key, value):
        self.filename = filename
        self.key = key
        self.value = value

    def exists(self):
        if not os.path.isfile(self.filename):
            return False
        try:
            with open(self.filename) as f:
                d = json.load(f)
                if not self.key in d:
                    return False
                return self.value == d[self.key]
        except:
            return False

class MongoDbCollectionTarget(luigi.Target):

    def __init__(self, db_name, db_host, collection, require_nonempty=False):

        self.db_name = db_name
        self.db_host = db_host
        self.collection = collection
        self.require_nonempty = require_nonempty

    def exists(self):

        logger.debug(
            "Host %s, DB %s, collection %s",
            self.db_host,
            self.db_name,
            self.collection)
        client = pymongo.MongoClient(self.db_host)
        db = client[self.db_name]

        exists = self.collection in db.list_collection_names()
        if not exists:
            logger.debug(
                "collection %s does NOT exist in %s",
                self.collection,
                self.db_name)
            return False

        if self.require_nonempty:

            empty = db[self.collection].count() == 0
            if empty:
                logger.debug(
                    "collection %s is EMPTY",
                    self.collection)

            return not empty

        else:

            return exists

class MongoDbDocumentTarget(luigi.Target):

    def __init__(self, db_name, db_host, collection, partial_document):

        self.db_name = db_name
        self.db_host = db_host
        self.collection = collection
        self.partial_document = partial_document

    def exists(self):

        logger.debug(
            "Host %s, DB %s, collection %s",
            self.db_host,
            self.db_name,
            self.collection)
        client = pymongo.MongoClient(self.db_host)
        db = client[self.db_name]

        exists = self.collection in db.list_collection_names()
        if not exists:
            logger.debug(
                "collection %s does NOT exist in %s",
                self.collection,
                self.db_name)
            return False

        collection = db[self.collection]

        return collection.count_documents(self.partial_document) > 0

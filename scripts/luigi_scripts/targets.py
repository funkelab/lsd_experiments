import luigi
import os
import z5py
import json
import pymongo

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
        if not os.path.isdir(self.filename):
            return False
        try:
            with z5py.File(self.filename, use_zarr_format=False, mode='r') as f:
                return self.dataset in f
        except:
            return False

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
        # print "Looking for %s:%s in %s"%(self.key,self.value,self.filename)
        if not os.path.isfile(self.filename):
            # print "%s does not exist"%self.filename
            return False
        try:
            with open(self.filename) as f:
                d = json.load(f)
                if not self.key in d:
                    # print "no key %s"%self.key
                    return False
                # print "%s == %s?"%(self.value,d[self.key])
                return self.value == d[self.key]
        except:
            return False

class MongoDbCollectionTarget(luigi.Target):

    def __init__(self, db_name, db_host, collection):

        self.db_name = db_name
        self.db_host = db_host
        self.collection = collection

    def exists(self):

        print("Host %s, DB %s, collection %s"%(self.db_host, self.db_name,
            self.collection))
        client = pymongo.MongoClient(self.db_host)
        db = client[self.db_name]

        return self.collection in db.list_collections()

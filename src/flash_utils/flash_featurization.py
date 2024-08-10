import os
from config import *
from provnet_utils import *

from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):

    def __init__(self, save_dir):
        self.epoch = 0
        self.save_dir = save_dir

    def on_epoch_end(self, model):
        model.save(os.path.join(self.save_dir,'word2vec.model'))
        self.epoch += 1

class EpochLogger(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        log("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        log("Epoch #{} end".format(self.epoch))
        self.epoch += 1
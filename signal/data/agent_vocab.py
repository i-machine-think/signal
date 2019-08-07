import os
import pickle

from helpers.file_helper import FileHelper

class AgentVocab(object):
    """
    Vocab object to create vocabulary and load if exists
    """

    START_TOKEN = "<S>"

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.file_helper = FileHelper()
        self.file_path = self.file_helper.get_vocabulary_path(self.vocab_size)
        if self.does_vocab_exist():
            self.load_vocab()
        else:
            self.build_vocab()

    def does_vocab_exist(self):
        return os.path.exists(self.file_path)

    def load_vocab(self):
        with open(self.file_path, "rb") as f:
            d = pickle.load(f)
            self.stoi = d["stoi"]  # dictionary w->i
            self.itos = d["itos"]  # list of words
            self.bound_idx = self.stoi[self.START_TOKEN]  # last word in vocab

    def save_vocab(self):
        with open(self.file_path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def build_vocab(self):
        self.stoi = {}
        self.itos = []

        for i in range(self.vocab_size - 1):
            self.itos.append(str(i))
            self.stoi[str(i)] = i

        self.itos.append(self.START_TOKEN)
        self.stoi[self.START_TOKEN] = len(self.itos) - 1
        self.bound_idx = self.stoi[self.START_TOKEN]
        self.save_vocab()

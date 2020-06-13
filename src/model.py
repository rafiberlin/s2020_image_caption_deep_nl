import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from collections import  Counter
import string
from torch.utils.data import Dataset
import pandas as pd

class ImageToHiddenState(nn.Module):
    """
    We try to transform each image to an hidden state with 120 values...
    """
    def __init__(self):
        super(ImageToHiddenState, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=3)
        self.out = nn.Linear(in_features=12 * 7 * 7, out_features=120)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=3, stride=3)

        t = t.reshape(-1, 12 * 7 * 7)
        t = self.out(t)
        t = F.relu(t)

        return t


class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None, mask_token="<MASK>", add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            mask_token (str): the MASK token to add into the Vocabulary; indicates
                a position that will not be used in updating the model's parameters
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary

        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token,
                'mask_token': self._mask_token}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary

        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    #use utf 8 otherwise bug
    with open(glove_filepath, "r",  encoding="utf8") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings

class CocoDatasetWrapper(Dataset):

    #TODO impose fixed length of the longest caption when vectorizing and test batch retrieval
    def __init__(self, cocodaset, vectorizer):
        self.cocodaset = cocodaset
        self.vectorizer = vectorizer

    def __len__(self):
        return self.cocodaset.__len__()

    def __getitem__(self, index):
        image, captions = self.cocodaset.__getitem__(index)
        # it seams like we always get 5 different captions for an image...
        for i, caption_reviewer in enumerate(captions):
                captions[i]["caption"] = self.vectorizer.vectorize(captions[i]["caption"])
        return image, captions

class CaptionVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, caption_vocab, max_sequence_length):
        self.caption_vocab = caption_vocab
        self.max_sequence_length = max_sequence_length

    def get_vocab(self):
        return self.caption_vocab

    def vectorize(self, title):
        """
        Args:
            title (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        indices = [self.caption_vocab.begin_seq_index]
        indices.extend(self.caption_vocab.lookup_token(token)
                       for token in title.split(" "))
        indices.append(self.caption_vocab.end_seq_index)

        # In a prediction task with added <start> and <end> tags,
        # if the label is: <start> it is a cat <end>
        # we should return the indexes for x_vector corresponding to the sequence
        # <start> it is a cat
        # and the indexes for y_vector corresponding to the sequence
        # it is a cat <end>
        # Multiplication by mask index insures that we are padding to the right for sequences
        # shorter than the max length caption in the data
        vector_length = self.max_sequence_length -1

        x_vector = np.ones(vector_length, dtype=np.int64) * self.caption_vocab.mask_index
        y_vector = np.ones(vector_length, dtype=np.int64) * self.caption_vocab.mask_index
        x_vector[:len(indices) - 1] = indices[:len(indices) - 1]
        y_vector[: len(indices) - 1] = indices[1:]
        return x_vector, y_vector

    @classmethod
    def from_dataframe(cls, captions, cutoff=5, exclude_punctuation=False):

        captions_frame = pd.DataFrame(captions, columns=["captions"])
        # +2 because we add the starting and ending sequence tags...
        max_sequence_length = max(map(len, captions_frame.captions)) + 2

        word_counts = Counter()
        for caption in captions:
            for token in caption.split(" "):
                if exclude_punctuation:
                    if token not in string.punctuation:
                        word_counts[token] += 1
                else:
                    word_counts[token] += 1

        caption_vocab = SequenceVocabulary()

        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                caption_vocab.add_token(word)

        return cls(caption_vocab, max_sequence_length)

    @classmethod
    def from_serializable(cls, contents):
        caption_vocab = SequenceVocabulary.from_serializable(contents)
        return cls(caption_vocab=caption_vocab)

    def to_serializable(self):
        return {'caption_vocab': self.caption_vocab.to_serializable()}

class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
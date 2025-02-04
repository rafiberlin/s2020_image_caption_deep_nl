import numpy as np
import pandas as pd
from collections import Counter
import torch
import string


class Vocabulary:
    """
    Class to process text and extract vocabulary for mapping
    Adapted from 'Natural Language Processing with PyTorch'
    """

    def __init__(self, token_to_idx=None, mask_token="<MASK>", add_unk=True, unk_token="<UNK>"):
        """
        Constructor

        :param token_to_idx: a pre-existing map of tokens to indices
        :param mask_token: the MASK token to add into the Vocabulary; indicates a position that will not be used in updating the model's parameters
        :param add_unk: a flag that indicates whether to add the UNK token
        :param unk_token: the UNK token to add into the Vocabulary
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

        :param token: The item to add into the Vocabulary
        :return: The integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality

        :param token: The token to look up
        :return: The index corresponding to the token
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index
        
        :param index: The index to look up
        :return: The token corresponding to the index
        :raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class CaptionVectorizer:
    """
    The Vectorizer which coordinates the Vocabularies and puts them to use
    Adapted from 'Natural Language Processing with PyTorch'
    """

    def __init__(self, caption_vocab, max_sequence_length):
        self.caption_vocab = caption_vocab
        self.max_sequence_length = max_sequence_length

    def get_vocab(self):
        return self.caption_vocab

    def decode(self, vectorized_input):
        """
        Pytorch array with list of indices
        :param vectorized_input: Input vector
        :return: Caption converted to str
        """
        return " ".join([self.caption_vocab._idx_to_token[i.item()] for i in vectorized_input
                         if i.item() not in
                         [self.caption_vocab.begin_seq_index, self.caption_vocab.mask_index,
                          self.caption_vocab.end_seq_index]
                         ])

    def vectorize(self, title):
        """
        Converts a caption string into a vector

        :param title: the string of words separated by a space
        :param vector_length: an argument for forcing the length of index vector
        :return: the vetorized title (numpy.array)
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
        vector_length = self.max_sequence_length - 1

        x_vector = np.ones(vector_length, dtype=np.int64) * \
                   self.caption_vocab.mask_index
        y_vector = np.ones(vector_length, dtype=np.int64) * \
                   self.caption_vocab.mask_index
        x_vector[:len(indices) - 1] = indices[:len(indices) - 1]
        y_vector[: len(indices) - 1] = indices[1:]
        return x_vector, y_vector

    def create_starting_sequence(self):
        """
        Creates pytorch array with a starting sequence token
        :return: Empty vector with start token
        """
        starting_token = torch.ones(
            self.max_sequence_length - 1, dtype=torch.long) * self.caption_vocab.mask_index
        starting_token[0] = self.caption_vocab.begin_seq_index
        return starting_token

    @classmethod
    def from_dataframe(cls, captions, cutoff=5, exclude_punctuation=False):
        """
        Creates a vectorizer object from a dataframe

        :param captions: captions in a dataframe
        :param cutoff: the limit to consider a word as known
        :param exclude_punctuation: boolean to decide whether the punctuation should be included or not
        :return: an insatnce of CaptionVectorizer
        """
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

    def to_serializable(self):
        return {'caption_vocab': self.caption_vocab.to_serializable()}


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super().__init__(token_to_idx)

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

        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality

        :param token: The token to look up
        :return: The index corresponding to the token
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

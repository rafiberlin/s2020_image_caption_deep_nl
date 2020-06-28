import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from collections import  Counter
import string

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from torch.utils.data import Dataset
import pandas as pd
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
class ImageToHiddenState(nn.Module):
    """
    We try to transform each image to an hidden state with 120 values...
    TODO: make the other parameters configurable like num channel kernel size, strides... it works only with 640 by 640 images now

    """
    def __init__(self, output_dim=120):
        super(ImageToHiddenState, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=3)
        self.out = nn.Linear(in_features=12 * 7 * 7, out_features=output_dim)

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


class LSTMModel(nn.Module):

    """
    Results: (NO sorting per length during training needed)

    epoch: 120, loss: 13.1952, train acc: 89.12%, dev acc: 75.33%
    Overall Learning Time 131.0737464
    test acc: 76.08%

    Used Params

    N_EPOCHS = 120
    LEARNING_RATE = 0.01
    REPORT_EVERY = 5
    EMBEDDING_DIM = 30
    HIDDEN_DIM = 20
    BATCH_SIZE = 150
    N_LAYERS = 1


    """

    # I added the padding index, as it is important to flag the index
    # that contains dummy information to speed up learning in embeddings
    def __init__(self,
                 embedding_dim,
                 character_set_size,
                 hidden_dim_rnn,
                 hidden_dim_cnn,
                 padding_idx=None,
                 rnn_layers=1,
                 pretrained_embeddings=None,
                 drop_out_prob=0.2
                 ):
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size
        self.rnn_layers = rnn_layers
        self.hidden_dim_rnn = hidden_dim_rnn
        self.n_classes = character_set_size
        # The output should be the same size as the hidden state size of RNN
        # but attention, if you change the value from 120 to something else,
        # you will probably need to adjsut the sizes of the kernels / stride in
        # ImageToHiddenState

        if pretrained_embeddings:
            # TODO HACK: improve this constructor api
            self.embeddings = pretrained_embeddings
            self.embedding_dim = pretrained_embeddings.embedding_dim
        else:
            self.embeddings = nn.Embedding(num_embeddings=self.character_set_size,
                                embedding_dim=self.embedding_dim, padding_idx=padding_idx)

        self.image_cnn = ImageToHiddenState(hidden_dim_cnn)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim_rnn, self.rnn_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim_rnn, self.n_classes)
        self.drop_layer = nn.Dropout(p=drop_out_prob)

    def forward(self, inputs):
        # WRITE CODE HERE

        imgs, labels = inputs
        current_device = str(imgs.device)
        batch_size = len(imgs)
        image_hidden = self.image_cnn(imgs)
        #Image hidden is used to init the hidden states of the lstm cells.
        # it must have the shape (number of layers *time number of direction) * batch size * hidden dim
        # size we just do 1 layer 1 direction, unsqueeze(0) is fine
        image_hidden = image_hidden.unsqueeze(dim=0)
        # when image_hidden needs to be provided for lstm,
        # we need to init the memory cell as well
        lstm_cell_initial_state = torch.zeros(image_hidden.shape , dtype=torch.float, device=current_device)
        embeds = self.embeddings(labels)
        # for a given sample, it "flattens" all the captions into the second dimension
        # we get from a 4 dimension shape: batch_size * number of captions * caption length * embdeing dimension
        # to a 3 dimension shape batch_size * (number of captions * caption length) * embdeing dimension
        embeds = embeds.reshape((batch_size,-1,self.embedding_dim))
        # Recommendation: use a single input for lstm layer (no special initialization of the hidden layer):
        #lstm_out, hidden = self.lstm(embeds, (image_hidden, lstm_cell_initial_state))
        lstm_out, hidden = self.lstm(embeds, (image_hidden, image_hidden))

        # WRITE MORE CODE HERE
        # hidden is a tuple. It looks like the first entry in hidden is the last hidden state,
        # the second entry the first hidden state
        classes = self.linear(self.drop_layer(lstm_out))

        # squeeze make out.shape to batch_size times num_classes
        out = F.log_softmax(classes, dim=2)
        return out

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

class CocoDatasetWrapper(Dataset):

    #TODO impose fixed length of the longest caption when vectorizing and test batch retrieval
    def __init__(self, cocodaset, vectorizer):
        self.cocodaset = cocodaset
        self.vectorizer = vectorizer

    @classmethod
    def transform_batch_for_training(cls, batch, device="cpu"):
        """

        :param batch:
        :return: a tuple of 3 element: the images, the in-vectorized captions and
                the out-vectorized captions for the loss function
        """
        return batch[0].to(device), batch[2][0].to(device), batch[2][1].to(device)


    def __len__(self):
        return self.cocodaset.__len__()

    def __getitem__(self, index):
        image, captions = self.cocodaset.__getitem__(index)
        # it seams like we always get 5 different captions for an image...
        num_captions = len(captions)
        # self.vectorizer.max_sequence_length - 1 because in label and out labels are shifted by 1 to match
        # for example, if the last real word in a caption is cat, the expected output caption is <end>...
        vectorized_captions_in = torch.zeros((num_captions, self.vectorizer.max_sequence_length - 1) , dtype=torch.long)
        vectorized_captions_out = torch.zeros((num_captions, self.vectorizer.max_sequence_length - 1) , dtype=torch.long)
        for i, caption_reviewer in enumerate(captions):
                c = self.vectorizer.vectorize(captions[i]["caption"])
                vectorized_captions_in[i], vectorized_captions_out[i] = tuple(map(torch.from_numpy, c))
        return image, captions, (vectorized_captions_in,vectorized_captions_out)

class CocoEvalBleuOnly(COCOEvalCap):
    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()


class CaptionVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, caption_vocab, max_sequence_length):
        self.caption_vocab = caption_vocab
        self.max_sequence_length = max_sequence_length

    def get_vocab(self):
        return self.caption_vocab

    def decode(self, vectorized_input):
        """
        Pytorch array with list of indices
        :param vectorized_input:
        :return:
        """
        return " ".join([self.caption_vocab._idx_to_token[i.item()] for i in vectorized_input
                if i.item() not in
                [self.caption_vocab.begin_seq_index, self.caption_vocab.mask_index, self.caption_vocab.end_seq_index]
                ])

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

    def create_starting_sequence(self):
        """
        Creates pytorch array with a starting sequence token
        :return:
        """
        starting_token = torch.ones(self.max_sequence_length - 1, dtype=torch.long) * self.caption_vocab.mask_index
        starting_token[0] = self.caption_vocab.begin_seq_index
        return starting_token

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

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

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
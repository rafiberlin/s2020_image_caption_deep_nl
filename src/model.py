import torch
from datetime import datetime
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from collections import Counter
import string
import torchvision.models as models
from itertools import combinations
import os
import torchvision.transforms as transforms
import torchvision.datasets as dset
import preprocessing as prep
import gensim
from torch.utils.data import Dataset
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from scipy.stats.mstats import gmean
from tqdm import tqdm

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

class VGG16Module(nn.Module):
    """

    Uses a conv layer to scale from 640x640x3 down to 230x230x3
    Passes that image a modified VGG16
    VGG16 outputs a linear layer with output_dim features

    """

    def __init__(self, output_dim):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=182, stride=2)
        self.linear = nn.Linear(4096, output_dim)

        # Remove last four layers of vgg16
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-4])

    def forward(self, img):
        y = self.conv(img)
        with torch.no_grad():
            y = self.vgg16(y)
        y = self.linear(y)
        return y

class MobileNetModule(nn.Module):
    """

    Uses a conv layer to scale from 640x640x3 down to 256x256x3
    Mobilenet outputs a linear layer with output_dim features

    """

    def __init__(self, output_dim):
        super().__init__()
        self.mobile = models.mobilenet_v2(pretrained=True)
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=130, stride=2)
        self.linear = nn.Linear(1000, output_dim)

    def forward(self, img):
        y = self.conv(img)
        with torch.no_grad():
            y = self.mobile(y)
        y = self.linear(y)
        return y

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
                 hidden_dim_rnn,
                 hidden_dim_cnn,
                 pretrained_embeddings,
                 rnn_layers=1,
                 cnn_model=None,
                 drop_out_prob=0.2
                 ):
        super(LSTMModel, self).__init__()
        self.embeddings = pretrained_embeddings
        self.embedding_dim = self.embeddings.embedding_dim
        self.vocabulary_size = self.embeddings.num_embeddings
        self.rnn_layers = rnn_layers
        self.hidden_dim_rnn = hidden_dim_rnn
        # The output should be the same size as the hidden state size of RNN
        # but attention, if you change the value from 120 to something else,
        # you will probably need to adjsut the sizes of the kernels / stride in
        # ImageToHiddenState
        self.n_classes = self.vocabulary_size

        if cnn_model == "vgg16":
            print("Using vgg16...")
            self.image_cnn = VGG16Module(hidden_dim_cnn)
        elif cnn_model == "mobilenet":
            print("Using mobilenet...")
            self.image_cnn = MobileNetModule(hidden_dim_cnn)
        else:
            print("Using default cnn...")
            self.image_cnn = ImageToHiddenState(hidden_dim_cnn)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim_rnn, self.rnn_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim_rnn, self.n_classes)
        self.drop_layer = nn.Dropout(p=drop_out_prob)

    def forward(self, inputs):
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
    def create_dataloader(cls, hparams, c_vectorizer, dataset_name="train2017", image_dir=None):
        train_file = hparams[dataset_name]
        if image_dir == None:
            image_dir = os.path.join(hparams['root'], train_file)
        else:
            image_dir = os.path.join(hparams['root'], image_dir)
        caption_file_path = prep.get_cleaned_captions_path(hparams, train_file)
        print("Image dir:", image_dir)
        print("Caption file path:", caption_file_path)

        # rgb_stats = prep.read_json_config(hparams["rgb_stats"])
        stats_rounding = hparams["rounding"]
        rgb_stats = {"mean": [0.31686973571777344, 0.30091845989227295, 0.27439242601394653],
                     "sd": [0.317791610956192, 0.307492196559906, 0.3042858839035034]}
        rgb_mean = tuple([round(m, stats_rounding) for m in rgb_stats["mean"]])
        rgb_sd = tuple([round(s, stats_rounding) for s in rgb_stats["mean"]])
        # TODO create a testing split, there is only training and val currently...
        coco_train_set = dset.CocoDetection(root=image_dir,
                                            annFile=caption_file_path,
                                            transform=transforms.Compose([prep.CenteringPad(),
                                                                          transforms.Resize((640, 640)),
                                                                          # transforms.CenterCrop(IMAGE_SIZE),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(rgb_mean, rgb_sd)])
                                            )

        coco_dataset_wrapper = CocoDatasetWrapper(coco_train_set, c_vectorizer)
        batch_size = hparams["batch_size"]
        train_loader = torch.utils.data.DataLoader(coco_dataset_wrapper, batch_size=batch_size)
        return train_loader

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
        # only use 5 captions to be able to use faster vectorized operations
        # avoid exceptions in the collate function in the fetch part of the dataloader
        return image, captions[:5], (vectorized_captions_in[:5],vectorized_captions_out[:5])

class BleuScorer(object):

    @classmethod
    def evaluate_gold(cls,hparams, train_loader, idx_break=-1):

        # NEVER do [{}]* 5!!!!
        # https://stackoverflow.com/questions/15835268/create-a-list-of-empty-dictionaries
        hypothesis = [{} for _ in range(5)]
        references = [{} for _ in range(5)]
        for idx, current_batch in tqdm(enumerate(train_loader)):
            imgs, \
            annotations, training_labels = current_batch
            for sample_idx, image_id in enumerate(annotations[0]["image_id"]):
                # create the list of all 4 captions out of 5. Because range(5) is ordered, the result is
                # deterministic...
                for c in list(combinations(range(5), 4)):
                    for hypothesis_idx in range(5):
                        if hypothesis_idx not in c:
                            hypothesis[hypothesis_idx][image_id.item()] = [annotations[hypothesis_idx]["caption"][sample_idx]]
                            references[hypothesis_idx][image_id.item()] = [annotations[annotation_idx]["caption"][sample_idx] for annotation_idx in list(c)]
            if idx == idx_break:
                # useful for debugging
                break

        scores = []
        for ref, hyp in list(zip(references, hypothesis)):
            scores.append(cls.calc_scores(ref, hyp))


        pd_score = pd.DataFrame(scores).mean()

        if hparams["save_eval_results"]:
            dt = datetime.now(tz=None)
            timestamp = dt.strftime(hparams["timestamp_prefix"])
            filepath = os.path.join(hparams["model_storage"], timestamp+"_bleu_gold.json")
            prep.create_json_config(pd_score.to_dict(), filepath)

        return pd_score

    @classmethod
    def evaluate(cls, hparams, train_loader, network_model, c_vectorizer, end_token_idx=3, idx_break=-1):
        # there is no other mthod to retrieve the current device on a model...
        device = next(network_model.parameters()).device
        hypothesis = {}
        references = {}
        for idx, current_batch in enumerate(train_loader):
            imgs, annotations, training_labels = current_batch
            for sample_idx, image_id in tqdm(enumerate(annotations[0]["image_id"])):
                _id = image_id.item()
                starting_token = c_vectorizer.create_starting_sequence().to(device)
                img = imgs[sample_idx].unsqueeze(dim=0).to(device)
                caption = starting_token.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                input_for_prediction = (img, caption)

                #TODO plug the beam search prediction
                predicted_label = predict_greedy(network_model, input_for_prediction, end_token_idx)
                current_hypothesis = c_vectorizer.decode(predicted_label[0][0])
                hypothesis[_id] = [current_hypothesis]
                # packs all 5 labels for one image with the corresponding image id
                references[_id] = [annotations[annotation_idx]["caption"][sample_idx] for annotation_idx in
                                               range(5)]
                if hparams["print_prediction"]:
                    print("\n#########################")
                    print("image", _id)
                    print("prediction", hypothesis[_id])
                    print("gold captions", references[_id])

            if idx == idx_break:
                # useful for debugging
                break
        score = cls.calc_scores(references, hypothesis)
        pd_score = pd.DataFrame([score])

        if hparams["save_eval_results"]:
            dt = datetime.now(tz=None)
            timestamp = dt.strftime(hparams["timestamp_prefix"])
            filepath = os.path.join(hparams["model_storage"], timestamp+"_bleu_prediction.json")
            filepath_2 = os.path.join(hparams["model_storage"], timestamp+"_bleu_prediction_scores.json")
            prep.create_json_config({ k:(hypothesis[k],references[k]) for k in hypothesis.keys()}, filepath)
            prep.create_json_config([score], filepath_2)

        """
        this code delivers the same results but we can't calculate a reasonable score on the 
        predefined labels to compare against gold (we always get 100%...)
        res looks like this : [{"image_id": 139, "caption": "a woman posing for the camera standing on skis"}, ... ]
        from pycocoevalcap.eval import COCOEvalCap
        coco_cap = COCO(caption_file_path)
        #imgIds = sorted([ batch_one[1][0]["image_id"][i].item() for i in range(batch_size)])
        imgIds = sorted([ id for id in hypothesis.keys()])
        res = [ {"image_id": k, "caption": hypothesis[k][0]} for k in sorted(hypothesis.keys())]
        prep.create_json_config(res, "./res_.json", 0)
        coco_res = coco_cap.loadRes("./res_.json")
        #CocoEvalBleuOnly is a copy Paste of the CocoEval class but where we only put BleuScorer...
        cocoEval = model.CocoEvalBleuOnly(coco_cap, coco_res)
        cocoEval.params['image_id'] = imgIds
        s = cocoEval.evaluate()
        """

        return pd_score

    @classmethod
    def calc_scores(cls, ref, hypo):

        """
        Code from https://www.programcreek.com/python/example/103421/pycocoevalcap.bleu.bleu.Bleu
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    @classmethod
    def perform_whole_evaluation(cls, hparams, loader, network, c_vectorizer, break_training_loop_idx=3):
        print("Run complete evaluation for:", loader.__repr__())
        train_bleu_score = BleuScorer.evaluate(hparams, loader, network, c_vectorizer,
                                                     idx_break=break_training_loop_idx)
        print("Unweighted Current Bleu Scores:\n", train_bleu_score)
        train_bleu_score_pd =train_bleu_score.to_numpy().reshape(-1)
        print("Weighted Current Bleu Scores:\n", train_bleu_score_pd.mean())
        print("Geometric Mean Current Bleu Score:\n", gmean(train_bleu_score_pd))
        bleu_score_human_average = BleuScorer.evaluate_gold(hparams, loader, idx_break=break_training_loop_idx)
        bleu_score_human_average_np = bleu_score_human_average.to_numpy().reshape(-1)
        print("Unweighted Gold Bleu Scores:\n", bleu_score_human_average)
        print("Weighted Gold Bleu Scores:\n", bleu_score_human_average_np.mean())
        print("Geometric Gold Bleu Scores:\n", gmean(bleu_score_human_average_np))


def predict_beam(model, input_for_prediction, end_token_idx, c_vectorizer, beam_width = 3):
    """
    WIP implementation of beam search
    """

    seq_len = 10#input_for_prediction[1].shape[2]
    device = next(model.parameters()).device

    image, vectorized_seq = input_for_prediction

    # first dimension 0 keeps indices, 1 keeps probability, 2 word corresponds to k-index of previous timestep
    track_best = torch.zeros((3, beam_width, seq_len)).to(device)
    model.eval()

    # Do first prediction, store #beam_width best
    pred = model(input_for_prediction)
    first_predicted = torch.topk(pred[0][0], beam_width)
    for i, (log_prob, index) in enumerate(zip(first_predicted.values, first_predicted.indices)):
        track_best[0,i,0] = index.item()
        track_best[1,i,0] = log_prob
        track_best[2,i,0] = -1

        #print(c_vectorizer.get_vocab().lookup_index(index.item()))

    #print("First:", first_predicted)        

    vocab_size = len(c_vectorizer.get_vocab())
    current_predictions = torch.zeros((beam_width * vocab_size))

    # For every sequence index consider all previous beam_width possibilities
    for idx in range(1, seq_len):
        for k in range(beam_width):
            i = track_best[0,k,idx-1]
            p = track_best[1,k,idx-1]

            # Build new sequence with previous index
            new_seq = vectorized_seq.detach().clone()
            new_seq[0][0][idx] = i

            # Predict new indices and rank beam_width best
            new_input = (image, new_seq)
            new_prediction = model(new_input)[0][idx] + p

            del new_seq

            # Store prediction
            current_predictions[k*vocab_size:(k+1)*vocab_size] = new_prediction[:]

        # Rank all predictions
        new_predicted = torch.topk(current_predictions, beam_width)

        # Find topk across all beam_width * vocab_size predictions
        for i, (log_prob, index) in enumerate(zip(new_predicted.values, new_predicted.indices)):
            # Find the correct word
            k_idx = index // vocab_size
            word_idx = index % vocab_size

            track_best[0,i,idx] = word_idx
            track_best[1,i,idx] = log_prob
            track_best[2,i,idx] = k_idx
            
            #print(idx, k_idx, c_vectorizer.get_vocab().lookup_index(word_idx.item()))

    # backtrack best result
    last_col = track_best[1,:,seq_len-1]
    best_k = torch.argmax(last_col, dim=0)
    #print("First best", best_k)
    
    indices = torch.zeros(seq_len)
    #indices[seq_len-1] = track_best[0,best_k, ]

    for idx in reversed(range(seq_len)):
        best_k = best_k.long()
        indices[idx] = track_best[0,best_k,idx]
        best_k = track_best[2,best_k,idx]

    #for index in indices:
    #    print(c_vectorizer.get_vocab().lookup_index(index.item()))

    #print(vectorized_seq.shape, indices.shape)

    # TODO: sample track_best and check if new_predicted works correctly
    return indices.unsqueeze(0).unsqueeze(0)

def predict_greedy(model, input_for_prediction, end_token_idx= 3 , prediction_number= 1, found_sequences = 0):
    """
    Only for dev purposes, allow us to get some outputs.
    :param model:
    :param input_for_prediction:
    :param end_token_idx:
    :param prediction_number:
    :param found_sequences:
    :return:
    """
    seq_len = input_for_prediction[1].shape[2]
    device = next(model.parameters()).device
    image, vectorized_seq = input_for_prediction
    # first dimension 0 keeps indices, 1 keeps probaility
    track_best = torch.zeros((2, prediction_number, seq_len)).to(device)
    model.eval()
    #TODO implement the whole sequence prediction using beam search...
    prediction_number = prediction_number - found_sequences
    for idx in range(seq_len - 1):
        pred = model(input_for_prediction)
        first_predicted = torch.topk(pred[0][idx], prediction_number)
        losses = first_predicted.values
        indices = first_predicted.indices
        idx_found_sequences = indices[indices == end_token_idx]
        found_sequences = idx_found_sequences.sum()
        vectorized_seq[0][0][idx+1] = indices[0]
        input_for_prediction = (image, vectorized_seq)
        if found_sequences > 0:
            break
    return vectorized_seq


def create_embedding(hparams, c_vectorizer, padding_idx=0):
    vocabulary_size = len(c_vectorizer.get_vocab())
    if(hparams["use_glove"]):
        print("Loading glove vectors...")
        glove_path = os.path.join(hparams['root'], hparams['glove_embedding'])
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)
        glove_embedding = np.zeros((vocabulary_size, glove_model.vector_size))
        token2idx = {word: glove_model.vocab[word].index for word in glove_model.vocab.keys()}
        for word in c_vectorizer.get_vocab()._token_to_idx.keys():
            i = c_vectorizer.get_vocab().lookup_token(word)
            if word in token2idx:
                glove_embedding[i, :] = glove_model.vectors[token2idx[word]]
            else:
                # From NLP with pytorch, it should be better to init the unknown tokens
                embedding_i = torch.ones(1, glove_model.vector_size)
                torch.nn.init.xavier_uniform_(embedding_i)
                glove_embedding[i, :] = embedding_i

        embed_size = glove_model.vector_size
        if hparams["embedding_dim"] < embed_size:
            embed_size = hparams["embedding_dim"]

        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embedding[:, :embed_size]))
        # TODO: not sure if we should not start with the pretrained but still be learning with our
        # training => transfer learning...
        embedding.weight.requires_grad =  hparams["improve_embedding"]
        print("GloVe embedding size:", glove_model.vector_size)
    else:
        nn.Embedding(num_embeddings=vocabulary_size,
                     embedding_dim=hparams["embedding_dim"], padding_idx=padding_idx)
    return embedding

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

def create_model_name(hparams):
    """
    Creates a model name that encodes the training parameters
    :param hparams:
    :return:
    """

    root_name, extension = hparams["model_name"].split(".")
    model_name = f"{hparams['cnn_model']}_{root_name}_cnn{str(hparams['hidden_dim_cnn'])}_rnn{str(hparams['hidden_dim_rnn'])}_emb{str(hparams['embedding_dim'])}_lr{str(hparams['lr'])}_ne{str(hparams['num_epochs'])}_bs{str(hparams['batch_size'])}_do{str(hparams['drop_out_prob'])}.{extension}"
        
    return model_name


def reminder_rnn_size():
    rnn_layer = 1
    feature_size = 30
    hidden_size = 20
    seq = 3
    batch_size = 5

    # self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim_rnn, self.rnn_layers, batch_first=True, dev)
    rnn = nn.LSTM(feature_size, hidden_size, rnn_layer, batch_first=True)
    input = torch.randn(batch_size, seq, feature_size)
    h0 = torch.randn(rnn_layer, batch_size, hidden_size)
    c0 = torch.randn(rnn_layer, batch_size, hidden_size)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output.shape)
    print(hn.shape)


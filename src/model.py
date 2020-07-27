import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import os
from tqdm import tqdm
from vocab import *


class ImageToHiddenState(nn.Module):
    """
    We try to transform each image to an hidden state with 120 values...
    DO NOT USE, broken!
    """

    def __init__(self, output_dim=120):
        super(ImageToHiddenState, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=12, kernel_size=5, stride=3)
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

    def __init__(self, output_dim, improve_pretrained=False):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.linear = nn.Linear(4096, output_dim)
        self.improve_pretrained = improve_pretrained
        # Remove last four layers of vgg16
        self.vgg16.classifier = nn.Sequential(
            *list(self.vgg16.classifier.children())[:-4])

    def forward(self, img):
        if not self.improve_pretrained:
            with torch.no_grad():
                y = self.vgg16(img)
            y = self.linear(y)
            # y = y.relu()
        else:
            y = self.vgg16(img)
            y = self.linear(y)
            # y = y.relu()
        return y


class Resnet50Module(nn.Module):

    def __init__(self, embed_size, improve_pretrained=False):
        super(Resnet50Module, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.improve_pretrained = improve_pretrained

    def forward(self, img):
        if not self.improve_pretrained:
            with torch.no_grad():
                y = self.resnet(img)
            y = y.view(y.size(0), -1)
            y = self.linear(y)
        else:
            y = self.resnet(img)
            y = y.view(y.size(0), -1)
            y = self.linear(y)
        return y


class MobileNetModule(nn.Module):
    """

    Uses a conv layer to scale from 640x640x3 down to 256x256x3
    Mobilenet outputs a linear layer with output_dim features

    """

    def __init__(self, output_dim, improve_pretrained=False):
        super().__init__()
        self.mobile = models.mobilenet_v2(pretrained=True)
        self.linear = nn.Linear(1000, output_dim)
        self.improve_pretrained = improve_pretrained

    def forward(self, img):
        if not self.improve_pretrained:
            with torch.no_grad():
                y = self.mobile(img)
            y = self.linear(y)
            # y = y.relu()
        else:
            y = self.mobile(img)
            y = self.linear(y)
            # y = y.relu()
        return y


class RNNModel(nn.Module):

    def __init__(self,
                 hidden_dim,
                 pretrained_embeddings,
                 rnn_layers=1,
                 cnn_model=None,
                 rnn_model="lstm",
                 drop_out_prob=0.2,
                 improve_cnn=False
                 ):

        super(RNNModel, self).__init__()
        self.embeddings = pretrained_embeddings
        self.embedding_dim = self.embeddings.embedding_dim
        self.vocabulary_size = self.embeddings.num_embeddings
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.rnn_model = rnn_model
        self.improve_cnn = improve_cnn
        self.n_classes = self.vocabulary_size
        self.drop_out_prob = drop_out_prob
        if cnn_model == "vgg16":
            print("Using vgg16...")
            self.image_cnn = VGG16Module(self.embedding_dim, self.improve_cnn)
        elif cnn_model == "mobilenet":
            print("Using mobilenet...")
            self.image_cnn = MobileNetModule(
                self.embedding_dim, self.improve_cnn)
        elif cnn_model == "resnet50":
            print("Using resnet50...")
            self.image_cnn = Resnet50Module(
                self.embedding_dim, self.improve_cnn)
        else:
            print("Using default cnn...")
            self.image_cnn = ImageToHiddenState(self.embedding_dim)
        if self.rnn_model == "gru":
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.rnn_layers, batch_first=True,
                              dropout=drop_out_prob)
        else:
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.rnn_layers, batch_first=True,
                               dropout=drop_out_prob)
        self.linear = nn.Linear(self.hidden_dim, self.n_classes)
        # self.drop_layer = nn.Dropout(p=drop_out_prob)

    def forward(self, imgs, labels):
        batch_size = imgs.shape[0]
        number_captions = labels.shape[1]
        image_hidden = self.image_cnn(imgs)

        # Transform the image hidden to shape batch size * number captions * 1 * embedding dimension
        image_hidden = image_hidden.unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, number_captions, 1, 1)
        embeds = self.embeddings(labels)

        # adds the image for each batch sample and caption as the first input of the sequence.
        # its output will be discarded at the return statement, we don't use it in the loss function
        embeds = torch.cat((image_hidden, embeds), dim=2)
        embeds = embeds.reshape(
            (batch_size * number_captions, -1, self.embedding_dim))
        lstm_out, _ = self.rnn(embeds)

        classes = self.linear(lstm_out)

        out = F.log_softmax(classes, dim=2)
        # remove the output of the first hidden state corresponding to the image output...
        return out[:, 1:, :]

    def predict_greedy(self, input_for_prediction, end_token_idx=3):
        """
        Only for dev purposes, allow us to get some outputs.
        :param model:
        :param input_for_prediction:
        :param end_token_idx:
        :param prediction_number:
        :param found_sequences:
        :return:
        """
        with torch.no_grad():
            self.eval()
            seq_len = input_for_prediction[1].shape[2]
            image, vectorized_seq = input_for_prediction
            prediction_number = 1
            for idx in range(seq_len - 1):
                pred = self(image, vectorized_seq)
                first_predicted = torch.topk(pred[0][idx], prediction_number)
                indices = first_predicted.indices
                idx_found_sequences = indices[indices == end_token_idx]
                found_sequences = idx_found_sequences.sum()
                vectorized_seq[0][0][idx + 1] = indices[0]
                if found_sequences > 0:
                    break
            return vectorized_seq

    def predict_beam(self, input_for_prediction, beam_width=3):
        """
        WIP implementation of beam search
        """
        with torch.no_grad():
            self.eval()
            seq_len = 30  # input_for_prediction[1].shape[2]
            device = next(self.parameters()).device
            image, vectorized_seq = input_for_prediction

            # 0 keeps indices
            # 1 keeps probability
            # 2 word corresponds to k-index of previous timestep
            track_best = torch.zeros((3, beam_width, seq_len)).to(device)

            # Do first prediction, store #beam_width best
            pred = self(*input_for_prediction)
            first_predicted = torch.topk(pred[0][0], beam_width)
            for i, (log_prob, index) in enumerate(zip(first_predicted.values, first_predicted.indices)):
                track_best[0, i, 0] = index.item()
                track_best[1, i, 0] = log_prob
                track_best[2, i, 0] = -1

            vocab_size = self.vocabulary_size

            current_predictions = torch.zeros((beam_width * vocab_size))
            new_seq = torch.zeros((beam_width, seq_len),
                                  dtype=torch.long).to(device)

            # Write start token
            for i in range(3):
                new_seq[i][0] = vectorized_seq[0][0][0]

            # For every sequence index consider all previous beam_width possibilities
            for idx in range(1, seq_len):
                for k in range(beam_width):
                    i = track_best[0, k, idx - 1]
                    p = track_best[1, k, idx - 1]
                    best_k = track_best[2, k, idx - 1].long()

                    # Build new sequence with previous index
                    new_seq[k][idx] = i

                    for o in reversed(range(1, idx)):
                        new_seq[k][o] = track_best[0, best_k, o - 1]
                        best_k = track_best[2, best_k, o - 1].long()

                    # Predict new indices and rank beam_width best
                    new_input = (image, new_seq[k].unsqueeze(0).unsqueeze(0))
                    new_prediction = self(*new_input)[0][idx] + p

                    # Store prediction
                    current_predictions[k *
                                        vocab_size:(k + 1) * vocab_size] = new_prediction[:]

                # Rank all predictions
                new_predicted = torch.topk(current_predictions, beam_width)

                # Find topk across all beam_width * vocab_size predictions
                for i, (log_prob, index) in enumerate(zip(new_predicted.values, new_predicted.indices)):
                    # Find the correct word
                    k_idx = index // vocab_size
                    word_idx = index % vocab_size

                    track_best[0, i, idx] = word_idx
                    track_best[1, i, idx] = log_prob
                    track_best[2, i, idx] = k_idx

            # Find best result
            last_col = track_best[1, :, seq_len - 1]
            best_k = torch.argmax(last_col, dim=0)

            return new_seq[best_k].unsqueeze(0).unsqueeze(0)

    def predict_greedy_sample(self, input_for_prediction, end_token_idx=3):
        """
        Only for dev purposes, allow us to get some outputs.
        :param self:
        :param input_for_prediction:
        :param end_token_idx:
        :param prediction_number:
        :param found_sequences:
        :return:
        """
        with torch.no_grad():
            self.eval()
            seq_len = input_for_prediction[1].shape[2]
            image, vectorized_seq = input_for_prediction
            prediction_number = 1
            for idx in range(seq_len - 1):
                pred = self(image, vectorized_seq)
                sampled_index = torch.multinomial(
                    torch.exp(pred[0][idx]), prediction_number)
                vectorized_seq[0][0][idx + 1] = sampled_index
                if sampled_index == end_token_idx:
                    break
            return vectorized_seq

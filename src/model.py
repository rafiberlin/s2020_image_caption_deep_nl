import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
                 improve_cnn=False,
                 teacher_forcing=True
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
        self.teacher_forcing = teacher_forcing
    def forward(self, imgs, labels):
        batch_size = imgs.shape[0]
        number_captions = labels.shape[1]
        seq_len = labels.shape[2]
        image_hidden = self.image_cnn(imgs)
        start_index = labels[0][0][0]
        device = next(self.parameters()).device

        # Transform the image hidden to shape batch size * number captions * 1 * embedding dimension
        image_hidden = image_hidden.unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, number_captions, 1, 1)
        if self.teacher_forcing:
            embeds = self.embeddings(labels)

            # adds the image for each batch sample and caption as the first input of the sequence.
            # its output will be discarded at the return statement, we don't use it in the loss function
            embeds = torch.cat((image_hidden, embeds), dim=2)
            embeds = embeds.reshape(
                (batch_size * number_captions, -1, self.embedding_dim))
            lstm_out, _ = self.rnn(embeds)
            classes = self.linear(lstm_out)
            out = F.log_softmax(classes, dim=2)
        else:
            image_hidden = image_hidden.reshape((batch_size * number_captions, -1, self.embedding_dim))
            lstm_out, _ = self.rnn(image_hidden)
            classes = self.linear(lstm_out)
            out = F.log_softmax(classes, dim=2)
            # first_predicted = torch.topk(out[:, 0], 1)
            # indices = first_predicted.indices
            indices = torch.ones(batch_size * number_captions, dtype=torch.long).unsqueeze(dim=1).to(device)*start_index
            predicted_embeds = self.embeddings(indices)
            for idx in range(1, seq_len+1):
                #lstm_out, _ = self.rnn(predicted_embeds[:,idx - 1,:].unsqueeze(dim=1))
                lstm_out, _ = self.rnn(predicted_embeds)
                classes = self.linear(lstm_out)
                out = torch.cat((out, F.log_softmax(classes, dim=2)), dim=1)
                first_predicted = torch.topk(out[:, idx], 1)
                indices = first_predicted.indices
                #predicted_embeds = torch.cat((predicted_embeds, self.embeddings(indices)), dim=1)
                predicted_embeds = self.embeddings(indices)

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

    def predict_beam_early_stop(self, input_for_prediction, beam_width=3):
        """
        Does the regular beam earch on full sequence but stops at the first ended sequence.
        No memory prblems.
        :param input_for_prediction:
        :param beam_width:
        :return:
        """
        with torch.no_grad():
            self.eval()
            seq_len = input_for_prediction[1].shape[2]
            device = next(self.parameters()).device
            # TODO set it as parameter, used to stop the search
            end_token_id = 3
            mask_idx = 0
            image, vectorized_seq = input_for_prediction
            start_token_idx = vectorized_seq[0, 0, 0]

            track_best = torch.zeros((beam_width, seq_len), dtype=torch.long).to(device)
            found = []
            track_best[:, 0] = start_token_idx
            pred = self(*input_for_prediction)
            first_predicted = torch.topk(pred[0][0], beam_width)
            track_best[:, 1] = first_predicted.indices
            last_candidates = {beam_row: (best_idx.item(), first_predicted.values[beam_row].item()) for
                               beam_row, best_idx
                               in enumerate(first_predicted.indices)}
            current_candidates = []
            # starts from 2 because track best has been initialized with <start> and word number 1 already
            del pred
            for seq_idx in range(2, seq_len):
                # Start Handle found sequences
                found_sequences = [beam_row for beam_row in last_candidates.keys() if
                                   last_candidates[beam_row][0] == end_token_id]
                beam_width -= len(found_sequences)
                for found_row in found_sequences:
                    loss = last_candidates[found_row][1]
                    found.append((loss, track_best[found_row]))
                    del last_candidates[found_row]
                # updates the number of candidates available
                last_candidates = {i: last_candidates[k] for i, k in enumerate(sorted(last_candidates.keys()))}
                # End Handle found sequences
                if found_sequences:
                    update = torch.zeros((beam_width, seq_len), dtype=torch.long).to(device)
                    update_idx = 0
                    for beam_row in range(track_best.shape[0]):
                        if beam_row not in found_sequences:
                            update[update_idx, :] = track_best[beam_row, :]
                            update_idx += 1
                    track_best = update
                    del update
                for beam_row in range(beam_width):
                    # reuse the vectorized sequence and updates indices at each pas to save memory
                    vectorized_seq[0][0][:seq_idx] = track_best[beam_row, : seq_idx]
                    # detach because we don't need gradienst...
                    temp_prediction = self(image, vectorized_seq)
                    current_prediction_idx = seq_idx - 1
                    # exclude mask index
                    top_temp_predict = torch.topk(temp_prediction[0][current_prediction_idx][mask_idx+1:], beam_width)
                    #all indices must be shifted by one...
                    top_indices = top_temp_predict.indices.clone() + 1
                    last_loss = last_candidates[beam_row][1]
                    current_candidates.extend(
                        (beam_row, best_idx.item(), top_temp_predict.values[pred_row].item() + last_loss) for
                        pred_row, best_idx in enumerate(top_indices))
                    del top_temp_predict
                    del temp_prediction
                    del top_indices
                current_candidates.sort(key=lambda x: x[2], reverse=True)
                current_seq_length = seq_idx + 1
                # Update the vector with the best candidates and keep track of them
                top_candidates = current_candidates[:beam_width]
                current_candidates.clear()
                track_temp = torch.zeros((beam_width, current_seq_length), dtype=torch.long).to(device)
                for idx, backtrack_info in enumerate(top_candidates):
                    last_row = backtrack_info[0]
                    track_temp[idx, :] = track_best[last_row, : current_seq_length].clone()
                    predicted_idx = backtrack_info[1]
                    track_temp[idx, -1] = predicted_idx
                track_best[:, :current_seq_length] = track_temp
                del track_temp
                last_candidates.clear()
                last_candidates.update(
                    {beam_row: (best_val[1], best_val[2]) for beam_row, best_val in enumerate(top_candidates)})

        # Either something was found or we take the top candidate sitting on the first row...
        if found:
            found.sort(key=lambda x: x[0], reverse=True)
            sentence = found[0][1].unsqueeze(0).unsqueeze(0)
        else:
            sentence = track_best[0].unsqueeze(0).unsqueeze(0)
        return sentence

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
        else:
            y = self.vgg16(img)
            y = self.linear(y)
        return y


class Resnet50Module(nn.Module):

    def __init__(self, embed_size, improve_pretrained=False):
        super().__init__()
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
        """
        :param output_dim: Output dimension of the linear layer
        :param improve_pretrained: Whether or not to improve the pretrained mobilenet model
        """

        super().__init__()
        self.mobile = models.mobilenet_v2(pretrained=True)
        self.linear = nn.Linear(1000, output_dim)
        self.improve_pretrained = improve_pretrained

    def forward(self, img):
        if not self.improve_pretrained:
            with torch.no_grad():
                y = self.mobile(img)
            y = self.linear(y)
        else:
            y = self.mobile(img)
            y = self.linear(y)
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

        super().__init__()
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
            print("Using default cnn: mobilenet")
            self.image_cnn = MobileNetModule(
                self.embedding_dim, self.improve_cnn)
        if self.rnn_model == "gru":
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.rnn_layers, batch_first=True,
                              dropout=drop_out_prob)
        else:
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.rnn_layers, batch_first=True,
                               dropout=drop_out_prob)
        self.linear = nn.Linear(self.hidden_dim, self.n_classes)
        self.teacher_forcing = teacher_forcing

    def forward(self, imgs, labels):
        batch_size = imgs.shape[0]
        seq_len = labels.shape[1]
        image_hidden = self.image_cnn(imgs)
        start_index = labels[0][0]
        device = next(self.parameters()).device

        # Transform the image hidden to shape batch size * number captions * 1 * embedding dimension
        image_hidden = image_hidden.unsqueeze(dim=1)

        # if self.teacher_forcing:
        #     teacher_force = random.random() < 0.5
        # else:
        #     teacher_force = False

        if self.teacher_forcing:
            embeds = self.embeddings(labels)

            # adds the image for each batch sample and caption as the first input of the sequence.
            # its output will be discarded at the return statement, we don't use it in the loss function
            embeds = torch.cat((image_hidden, embeds), dim=1)
            lstm_out, _ = self.rnn(embeds)
            classes = self.linear(lstm_out)
            out = F.log_softmax(classes, dim=2)
        else:
            lstm_out, hidden = self.rnn(image_hidden)
            classes = self.linear(lstm_out)
            out = F.log_softmax(classes, dim=2)
            # first_predicted = torch.topk(out[:, 0], 1)
            # indices = first_predicted.indices
            indices = torch.ones(batch_size, dtype=torch.long).unsqueeze(dim=1).to(device)*start_index
            predicted_embeds = self.embeddings(indices)
            for idx in range(1, seq_len+1):
                lstm_out, hidden = self.rnn(predicted_embeds, hidden)
                classes = self.linear(lstm_out)
                out = torch.cat((out, F.log_softmax(classes, dim=2)), dim=1)
                #out = torch.cat((out, classes), dim=1)
                _, indices = torch.topk(out[:, idx], 1)
                #Sets everything to zero if the last index was end or mask
                indices[torch.nonzero((torch.topk(out[:, idx - 1], 1).indices == 3), as_tuple=False)[:, 0]] = 0
                indices[torch.nonzero((torch.topk(out[:, idx - 1], 1).indices == 0), as_tuple=False)[:, 0]] = 0
                predicted_embeds = self.embeddings(indices)
                del indices
                del lstm_out
                del classes

        # remove the output of the first hidden state corresponding to the image output...
        return out[:, 1:, :]

    def predict_greedy(self, input_for_prediction, end_token_idx=3):
        """
        Generate captions by selecting the highest probability word at each step

        :param input_for_prediction: a tuple containing the image and an empty sequence except
        for the <BEGIN> token
        :param end_token_idx: id of the vectorized <END> token
        :return: a vectorized sequence to be decoded
        """
        with torch.no_grad():
            self.eval()
            image, vectorized_seq = input_for_prediction
            batch_size = image.shape[0]
            seq_len = vectorized_seq.shape[1]
            image_hidden = self.image_cnn(image)
            start_index = vectorized_seq[0][0]
            device = next(self.parameters()).device
            image_hidden = image_hidden.unsqueeze(dim=1)

            lstm_out, hidden = self.rnn(image_hidden)
            classes = self.linear(lstm_out)
            out = F.log_softmax(classes, dim=2)
            indices = torch.ones(batch_size, dtype=torch.long).unsqueeze(dim=1).to(device)*start_index
            predicted_embeds = self.embeddings(indices)

            for idx in range(1, seq_len):
                lstm_out, hidden = self.rnn(predicted_embeds, hidden)
                classes = self.linear(lstm_out)
                out = torch.cat((out, F.log_softmax(classes, dim=2)), dim=1)
                _, indices = torch.topk(out[:, idx], 1)
                predicted_embeds = self.embeddings(indices)
                idx_found_sequences = indices[indices == end_token_idx]
                found_sequences = idx_found_sequences.sum()
                vectorized_seq[0][idx] = indices[0]
                if found_sequences > 0:
                    break
                del indices
                del lstm_out
                del classes

            return vectorized_seq

    def predict_beam_early_stop(self, input_for_prediction, beam_width=3):
        """
        Does the regular beam search on full sequence but stops at the first ended sequence.
        No memory problems.

        :param input_for_prediction: a tuple containing the image and an empty sequence except for the <BEGIN> token
        :param beam_width: Number of generated sequences to consider.
        :return: a vectorized sequence to be decoded
        """
        with torch.no_grad():
            self.eval()
            seq_len = input_for_prediction[1].shape[1]
            device = next(self.parameters()).device
            # TODO set it as parameter, used to stop the search
            end_token_id = 3
            mask_idx = 0
            image, vectorized_seq = input_for_prediction
            start_token_idx = vectorized_seq[0, 0]

            track_best = torch.zeros((beam_width, seq_len), dtype=torch.long).to(device)
            found = []
            track_best[:, 0] = start_token_idx
            pred = self(*input_for_prediction)
            first_predicted_val,  first_predicted_indices = torch.topk(pred[0][0], beam_width)
            track_best[:, 1] = first_predicted_indices
            last_candidates = {beam_row: (best_idx.item(), first_predicted_val[beam_row].item()) for
                               beam_row, best_idx
                               in enumerate(first_predicted_indices)}
            current_candidates = []
            # starts from 2 because track best has been initialized with <start> and word number 1 already
            del pred
            del first_predicted_val
            del first_predicted_indices
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
                    vectorized_seq[0][:seq_idx] = track_best[beam_row, : seq_idx]
                    # detach because we don't need gradienst...
                    temp_prediction = self(image, vectorized_seq)
                    current_prediction_idx = seq_idx - 1
                    # exclude mask index
                    top_indices_val, top_indices = torch.topk(temp_prediction[0][current_prediction_idx][mask_idx+1:], beam_width)
                    #all indices must be shifted by one...
                    top_indices = top_indices + 1
                    last_loss = last_candidates[beam_row][1]
                    current_candidates.extend(
                        (beam_row, best_idx.item(), top_indices_val[pred_row].item() + last_loss) for
                        pred_row, best_idx in enumerate(top_indices))
                    del top_indices_val
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
            sentence = found[0][1].unsqueeze(0)
        else:
            sentence = track_best[0].unsqueeze(0)
        return sentence


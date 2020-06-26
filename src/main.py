import torch
import torchvision.transforms as transforms
import torch.utils.data
import os
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from timeit import default_timer as timer
import gensim
# own modules
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

import model
import preprocessing as prep

HYPER_PARAMETER_CONFIG = "../hparams.json"
REPORT_EVERY = 5
EMBEDDING_DIM = 60
BATCH_SIZE = 150
PADDING_WORD = "<MASK>"
BEGIN_WORD = "<BEGIN>"
END_WORD = "<END>"
IMAGE_SIZE = 320

def main():
    hparams = prep.read_json_config(HYPER_PARAMETER_CONFIG)
    device = hparams["device"]
    if not torch.cuda.is_available():
        device = "cpu"

    cleaned_captions = prep.create_list_of_captions_and_clean(hparams, "train")
    embedding = None
    c_vectorizer = None
    vocabulary_size = 0

    if hparams["use_glove"]:
        ## TODO: pass embedding to model and reshape lstm input/output accordingly
        print("Loading glove vectors...")
        glove_path = os.path.join(hparams['root'], hparams['glove_embedding'])
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)
        c_vectorizer = model.GloVeVectorizer.from_dataframe(glove_model, cleaned_captions)

        # Add additional weights for sequence markers
        glove_add = np.random.normal(scale=0.6, size=(4, glove_model.vector_size))
        glove_weights = np.concatenate((glove_model.vectors, glove_add))

        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_weights))
        embedding.weight.requires_grad = False

        print("GloVe embedding size:", glove_model.vector_size)
        print("GloVe num embeddings:", len(glove_model.vocab))
        print("GloVe final weights shape:", glove_weights.shape)

        vocabulary_size = len(c_vectorizer.get_target_vocab())
    else:
        c_vectorizer = model.CaptionVectorizer.from_dataframe(cleaned_captions)
        vocabulary_size = len(c_vectorizer.get_vocab())

    #padding_idx = c_vectorizer.get_vocab()._token_to_idx[PADDING_WORD]
    image_dir = os.path.join(hparams['root'], hparams['train'])

    caption_file_path = prep.get_cleaned_captions_path(hparams, hparams['train'])
    print("Image dir:", image_dir)
    print("Caption file path:", caption_file_path)

    #rgb_stats = prep.read_json_config(hparams["rgb_stats"])
    stats_rounding = hparams["rounding"]
    rgb_stats ={"mean": [0.31686973571777344, 0.30091845989227295, 0.27439242601394653],     "sd": [0.317791610956192, 0.307492196559906, 0.3042858839035034]}
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

    coco_dataset_wrapper = model.CocoDatasetWrapper(coco_train_set, c_vectorizer)
    batch_size = hparams["batch_size"][0]
    train_loader = torch.utils.data.DataLoader(coco_dataset_wrapper, batch_size)

    ## Generate output folder if non-existent
    model_dir = hparams["model_storage"]
    if not os.path.isdir(model_dir):
        try:
            os.mkdir(model_dir)
        except OSError:
            print(f"Creation of the directory {model_dir} failed")
    model_path = os.path.join(model_dir, hparams["model_name"])
    print("Model save path:", model_path)

    ## Training start
    network = model.LSTMModel(EMBEDDING_DIM, vocabulary_size, hparams["hidden_dim_rnn"], hparams["hidden_dim_cnn"], padding_idx=None, pretrained_embeddings=embedding).to(device)
    #network = model.LSTMModel(EMBEDDING_DIM, vocabulary_size, HIDDEN_DIM_RNN, HIDDEN_DIM_CNN, pretrained_embeddings=embeddings).to(device)
    start_training = True
    if os.path.isfile(model_path):
        network.load_state_dict(torch.load(model_path))
        start_training = False
        print("Skip Training")
    else:
        print("Start Training")
    batch_one = next(iter(train_loader))
    if start_training:
        loss_function = nn.NLLLoss().to(device)
        optimizer = optim.Adam(params=network.parameters(), lr=hparams['lr'])
        start = timer()
        # --- training loop ---
        torch.cuda.empty_cache()
        network.train()
        total_loss = 0
        for epoch in range(hparams["num_epochs"]):
            # TODO build a bigger loop...
            images, in_captions, out_captions = model.CocoDatasetWrapper.transform_batch_for_training(batch_one, device)

            optimizer.zero_grad()
            # flatten all caption , flatten all batch and sequences, to make its category comparable
            # for the loss function
            out_captions = out_captions.reshape(-1)
            log_prediction = network((images, in_captions)).reshape(out_captions.shape[0], -1)
            # Warning if we are unable to learn, use the contiguus function of the tensor
            # it insures that the sequence is not messed up during reshape
            loss = loss_function(log_prediction, out_captions)
            loss.backward()
            print("Loss:", loss.item())
            # step 5. use optimizer to take gradient step
            optimizer.step()

        end = timer()
        print("Overall Learning Time", end - start)
        torch.save(network.state_dict(), model_path)

    images, in_captions, out_captions = model.CocoDatasetWrapper.transform_batch_for_training(batch_one, device)
    print_some_predictions(c_vectorizer, network, batch_size, device, images, in_captions)
    test_eval_api(c_vectorizer, network, batch_size, device, images, in_captions, batch_one)

def test_eval_api(c_vectorizer, network, batch_size, device, images, in_captions, batch_one):
    """
    Demonstrates of the eval API is working.
    :param c_vectorizer:
    :param network:
    :param batch_size:
    :param device:
    :param images:
    :param in_captions:
    :param batch_one:
    :return:
    """

    hyp = {}
    ref = {}
    for idx in range(batch_size):
        starting_token = c_vectorizer.create_starting_sequence().to(device)
        input_for_prediction = (images[idx].unsqueeze(dim=0), starting_token.unsqueeze(dim=0).unsqueeze(dim=0))
        predicted_label = predict_greedy(network, input_for_prediction, device)
        label = []
        for c in predicted_label[0][0]:
            l = c_vectorizer.get_vocab()._idx_to_token[c.item()]
            if l != END_WORD and l != BEGIN_WORD:
                label.append(l)
            if l == END_WORD:
                break
            if l == PADDING_WORD:
                break
        h = " ".join(label)
        for c_idx in range(5):
            label.clear()
            for c in in_captions[idx][c_idx]:
                l = c_vectorizer.get_vocab()._idx_to_token[c.item()]
                if l != END_WORD and l != BEGIN_WORD and l != PADDING_WORD:
                    label.append(l)
                if l == END_WORD:
                    break
                if l == PADDING_WORD:
                    break
            r = " ".join(label)
            id = batch_one[1][c_idx]["id"][idx]
            ref[id]=[r]
            hyp[id] = [h]

    scores = calc_scores(ref, ref)
    print("Best Possible Score:", scores)

    scores = calc_scores(ref, hyp)
    print("Our Score:", scores)

def print_some_predictions(c_vectorizer, network, batch_size, device, images, in_captions):
    #TODO Model predicts kind of weird stuff. Changin the init state of the LSTM cell
    # to the image and increasing the embedding size helped, but we need to observe that...
    # Moreover, argmax is not the best => we should sample the words from the prob distribution implied by
    # the predictions...
    for idx in range(batch_size):
        starting_token = c_vectorizer.create_starting_sequence().to(device)
        input_for_prediction = (images[idx].unsqueeze(dim=0), starting_token.unsqueeze(dim=0).unsqueeze(dim=0))
        predicted_label = predict_greedy(network, input_for_prediction, device)
        label = []
        for c in predicted_label[0][0]:
            l = c_vectorizer.get_vocab()._idx_to_token[c.item()]
            if l != END_WORD and l != BEGIN_WORD:
                label.append(l)
            if l == END_WORD:
                break
            if l == PADDING_WORD:
                break
        print("##################################\n")
        print("predicted label:", " ".join(label))
        for c_idx in range(5):
            label.clear()
            for c in in_captions[idx][c_idx]:
                l = c_vectorizer.get_vocab()._idx_to_token[c.item()]
                if l != END_WORD and l != BEGIN_WORD:
                    label.append(l)
                if l == END_WORD:
                    break
                if l == PADDING_WORD:
                    break
            print("real label:", " ".join(label))



def calc_scores(ref, hypo):

    """
    Code from https://www.programcreek.com/python/example/103421/pycocoevalcap.bleu.bleu.Bleu
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
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



def predict_greedy(model, input_for_prediction, device, prediction_number= 1, found_sequences = 0, end_token_idx= 3):
    seq_len = input_for_prediction[1].shape[2]

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

def predict(model, input_for_prediction, device, prediction_number= 3, found_sequences = 0, end_token_idx= 3):
    seq_len = input_for_prediction[1].shape[2]
    # first dimension 0 keeps indices, 1 keeps probaility
    track_best = torch.zeros((2, prediction_number, seq_len)).to(device)
    model.eval()
    #TODO implement the whole sequence prediction using beam search...
    prediction_number = prediction_number - found_sequences
    for idx in range(seq_len):
        pred = model(input_for_prediction)
        first_predicted = torch.topk(pred[0][idx], prediction_number)
        losses = first_predicted.values
        indices = first_predicted.indices
        idx_found_sequences = indices[indices == end_token_idx]
        found_sequences = idx_found_sequences.sum()
        if found_sequences >= prediction_number:
            break
        #predicted_word = c_vectorizer.get_vocab()._idx_to_token[]
        pass
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


if __name__ == '__main__':
    main()
    # reminder_rnn_size()

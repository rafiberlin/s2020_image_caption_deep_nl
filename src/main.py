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
from pycocotools.coco import COCO
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import pandas as pd
from itertools import combinations
import model
import preprocessing as prep
import argparse

HYPER_PARAMETER_CONFIG = "hparams.json"
EMBEDDING_DIM = 60
PADDING_WORD = "<MASK>"
BEGIN_WORD = "<BEGIN>"
END_WORD = "<END>"
IMAGE_SIZE = 320

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", help="hparams config file")
    parser.add_argument("--train", action="store_true", help="force training")
    args = parser.parse_args()

    if args.params:
        hparams = prep.read_json_config(args.params)
    else:
        hparams = prep.read_json_config(HYPER_PARAMETER_CONFIG)

    device = hparams["device"]
    if not torch.cuda.is_available():
        device = "cpu"

    cleaned_captions = prep.create_list_of_captions_and_clean(hparams, "train")
    embedding = None
    c_vectorizer = model.CaptionVectorizer.from_dataframe(cleaned_captions)
    vocabulary_size = len(c_vectorizer.get_vocab())

    if hparams["use_glove"]:
        print("Loading glove vectors...")
        glove_path = os.path.join(hparams['root'], hparams['glove_embedding'])
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)

        glove_embedding = np.random.rand(vocabulary_size, glove_model.vector_size)

        token2idx = {}
        for word in glove_model.vocab.keys():
            token2idx[word] = glove_model.vocab[word].index

        for word in c_vectorizer.get_vocab()._token_to_idx.keys():
            i = c_vectorizer.get_vocab().lookup_token(word)
            if word in token2idx:
                glove_embedding[i, :] = glove_model.vectors[token2idx[word]]

        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embedding))
        embedding.weight.requires_grad = False
        print("GloVe embedding size:", glove_model.vector_size)
        

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
    network = model.LSTMModel(EMBEDDING_DIM, vocabulary_size, hparams["hidden_dim_rnn"], hparams["hidden_dim_cnn"], padding_idx=None, pretrained_embeddings=embedding, cnn_model=hparams["cnn_model"]).to(device)
    #network = model.LSTMModel(EMBEDDING_DIM, vocabulary_size, HIDDEN_DIM_RNN, HIDDEN_DIM_CNN, pretrained_embeddings=embeddings).to(device)
    start_training = True
    if os.path.isfile(model_path) and not args.train:
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
    create_coco_eval_frame(c_vectorizer, network, batch_size, device, images, in_captions, batch_one)
    """
    from pycocoevalcap.eval import COCOEvalCap
    coco_cap = COCO(caption_file_path)
    res = create_coco_eval_frame(c_vectorizer, network, batch_size, device, images, in_captions, batch_one)
    imgIds = sorted([ batch_one[1][0]["image_id"][i].item() for i in range(batch_size)])
    prep.create_json_config(res, "./res.json", 0)
    coco_res = coco_cap.loadRes("./res.json")
    cocoEval = model.CocoEvalBleuOnly(coco_cap, coco_res)
    cocoEval.params['image_id'] = imgIds
    s = cocoEval.evaluate()
    labs = get_original_label(batch_one, batch_size)
    for i, l in enumerate(labs):
        print("ref", i)
        prep.create_json_config(labs[i], f"./ref_{i}.json")
        eval = model.CocoEvalBleuOnly(coco_cap, coco_cap.loadRes(f"./ref_{i}.json"))
        eval.params['image_id'] = imgIds
        eval.evaluate()
    pass
    """
    labs = get_original_label(batch_one, batch_size)
    print_some_predictions(c_vectorizer, network, batch_size, device, images, in_captions)
    #test_eval_api(c_vectorizer, network, batch_size, device, images, in_captions, batch_one)
    #score_list = calculate_ref_score_for_bleu(c_vectorizer, network, batch_size, device, images, in_captions, batch_one)
    #print(score_list)

def create_coco_eval_frame(c_vectorizer, network, batch_size, device, images, in_captions, batch_one):
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

    total_captions = 5
    score_list = []
    for idx in range(batch_size):
        starting_token = c_vectorizer.create_starting_sequence().to(device)
        input_for_prediction = (images[idx].unsqueeze(dim=0), starting_token.unsqueeze(dim=0).unsqueeze(dim=0))
        predicted_label = predict_greedy(network, input_for_prediction, device)
        current_hypothesis = c_vectorizer.decode(predicted_label[0][0])
        id = batch_one[1][0]["image_id"][idx].item()
        score_list.append({"image_id": id, "caption": current_hypothesis})

    return score_list

def get_original_label(batch_one, batch_size):
    total_captions = 5
    ref = None
    ref_list = [ref] * total_captions
    for idx in range(total_captions):
        ref_list[idx] = [{"image_id": id.item(), "caption": batch_one[1][idx]["caption"][i]} for i, id in enumerate(batch_one[1][idx]["image_id"]) ]

    return ref_list

def get_original_label(batch_one, batch_size):
    total_captions = 5
    ref = None
    ref_list = [ref] * total_captions
    for idx in range(total_captions):
        id = batch_one[1][idx]["id"][0].item()
        caption = batch_one[1][idx]["caption"][0]
    return ref_list

def calculate_ref_score_for_bleu(c_vectorizer, network, batch_size, device, images, in_captions, batch_one):
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

    total_captions = 5
    hyp = []
    ref = []
    hyp_list = [hyp] * total_captions
    ref_list = [ref] * total_captions
    score_list = []
    for idx in range(batch_size):
        label = []
        label_range = range(total_captions)
        for c_idx in label_range:
            label.clear()
            current_hypothesis = c_vectorizer.decode(in_captions[idx][c_idx])
            list_ref = []
            for c_idx_ref in label_range:
                if c_idx_ref != c_idx:
                    current_ref = c_vectorizer.decode(in_captions[idx][c_idx_ref])
                    list_ref.append(current_ref)
            id = batch_one[1][0]["id"][idx]
            ref_list[c_idx][id].append({"image_id": id, "caption": current_hypothesis})


    df_score = pd.DataFrame(score_list)
    return df_score

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
            l = c_vectorizer.caption_vocab._idx_to_token[c.item()]
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
                l = c_vectorizer.caption_vocab._idx_to_token[c.item()]
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
        predicted_label = predict_beam(network, input_for_prediction, device, c_vectorizer)
        label = []
        for c in predicted_label[0][0]:
            l = c_vectorizer.caption_vocab._idx_to_token[c.item()]
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
                l = c_vectorizer.caption_vocab._idx_to_token[c.item()]
                if l != END_WORD and l != BEGIN_WORD and l != PADDING_WORD:
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

def predict_beam(model, input_for_prediction, device, c_vectorizer, beam_width = 3, found_sequences = 0, end_token_idx= 3):
    seq_len = input_for_prediction[1].shape[2]
    image, vectorized_seq = input_for_prediction

    print("Input seq:", vectorized_seq.shape)

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

        print(c_vectorizer.get_vocab().lookup_index(index.item()))

    print("First:", first_predicted)        

    vocab_size = len(c_vectorizer.get_vocab())
    current_predictions = torch.zeros((beam_width * vocab_size))

    # For every sequence index consider all previous beam_width possibilities
    for idx in range(1, 10):
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
            
            print(idx, k_idx, c_vectorizer.get_vocab().lookup_index(word_idx.item()))

    return vectorized_seq

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

    hypothesis = 'that is a meal'

    ref = ['closeup of bins of food that include broccoli and bread',
    'a meal is presented in brightly colored plastic trays',
    'there are containers filled with different kinds of foods',
    'colorful dishes holding meat vegetables fruit and bread',
    'a bunch of trays that have different food']

    comb = combinations(ref, 4)
    for c in list(comb):
        for r in ref:
            if r not in c:
                h = {}
                rr = {}
                for i, e in enumerate (c):
                    h[i]=[r]
                    rr[i]=[e]
                calc_scores(rr, h)

        # reminder_rnn_size()

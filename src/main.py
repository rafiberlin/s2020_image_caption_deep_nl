#!/usr/bin/env python3
import torch
import torchvision.transforms as transforms
import torch.utils.data
import os
import torchvision.datasets as dset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timeit import default_timer as timer
import gensim
import model
import preprocessing as prep
import argparse

HYPER_PARAMETER_CONFIG = "./hparams.json"
EMBEDDING_DIM = 60
PADDING_WORD = "<MASK>"
BEGIN_WORD = "<BEGIN>"
END_WORD = "<END>"
IMAGE_SIZE = 320

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", help="hparams config file")
    parser.add_argument("--train", action="store_true", help="force training")
    parser.add_argument("--download", action="store_true", help="download dataset")
    args = parser.parse_args()

    if args.params:
        hparams = prep.read_json_config(args.params)
    else:
        hparams = prep.read_json_config(HYPER_PARAMETER_CONFIG)

    if args.download:
        prep.download_images(hparams["img_train_url"], "./data")
        prep.download_images(hparams["img_val_url"], "./data")
    train_file = hparams['val']
    device = hparams["device"]
    if not torch.cuda.is_available():
        device = "cpu"
    cleaned_captions = prep.create_list_of_captions_and_clean(hparams, "val")
    #cleaned_captions = prep.create_list_of_captions_and_clean(hparams, "train")
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
    image_dir = os.path.join(hparams['root'], train_file)

    caption_file_path = prep.get_cleaned_captions_path(hparams, train_file)
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

    bleu_score = model.BleuScorer.evaluate(train_loader, network, c_vectorizer, idx_break=3, print_prediction=True)
    print("Unweighted Current Bleu Scores", bleu_score)
    print("Weighted Current Bleu Scores", bleu_score.mean())
    bleu_score_human_average = model.BleuScorer.evaluate_gold(train_loader, idx_break=3)
    print("Unweighted Gold Bleu Scores", bleu_score)
    print("Weighted Gold Bleu Scores", bleu_score.mean())

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

    #calculate_avg_bleu_human()

    # reminder_rnn_size()

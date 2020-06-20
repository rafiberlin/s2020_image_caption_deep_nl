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
import model
import preprocessing as prep

HYPER_PARAMETER_CONFIG = "hparams.json"
REPORT_EVERY = 5
EMBEDDING_DIM = 30
HIDDEN_DIM = 20
HIDDEN_DIM_CNN = 100
HIDDEN_DIM_RNN = 100
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
    c_vectorizer = model.CaptionVectorizer.from_dataframe(cleaned_captions)
    padding_idx = c_vectorizer.caption_vocab._token_to_idx[PADDING_WORD]

    if hparams["use_glove"]:
        ## TODO: pass embedding to model
        print("Loading glove vectors...")
        glove_path = os.path.join(hparams['root'], hparams['glove_embedding'])
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)
        glove_weights = torch.FloatTensor(glove_model.vectors)
        embedding = nn.Embedding.from_pretrained(glove_weights)
        embedding.weight.requires_grad = False

    image_dir = os.path.join(hparams['root'], hparams['train'])

    caption_file_path = prep.get_captions_path(hparams, hparams['train'])
    print("Image dir:", image_dir)
    print("Caption file path:", caption_file_path)

    #rgb_stats = prep.read_json_config(hparams["rgb_stats"])
    #stats_rounding = hyper_parameters["rounding"]
    #rgb_mean = tuple([round(m, stats_rounding) for m in rgb_stats["mean"]])
    #rgb_sd = tuple([round(s, stats_rounding) for s in rgb_stats["mean"]])
    # TODO create a testing split, there is only training and val currently...
    coco_train_set = dset.CocoDetection(root=image_dir,
                                        annFile=caption_file_path,
                                        transform=transforms.Compose([prep.CenteringPad(),
                                                                      transforms.Resize((640, 640)),
                                                                      # transforms.CenterCrop(IMAGE_SIZE),
                                                                      transforms.ToTensor()])
                                                                      #transforms.Normalize(rgb_mean, rgb_sd)])
                                        )

    coco_dataset_wrapper = model.CocoDatasetWrapper(coco_train_set, c_vectorizer)
    batch_size = hparams["batch_size"][0]
    train_loader = torch.utils.data.DataLoader(coco_dataset_wrapper, batch_size)

    vocabulary_size = len(c_vectorizer.caption_vocab)

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
    network = model.LSTMModel(EMBEDDING_DIM, vocabulary_size, HIDDEN_DIM_RNN, HIDDEN_DIM_CNN, padding_idx).to(device)
    start_training = True
    if os.path.isfile(model_path):
        network.load_state_dict(torch.load(model_path))
        start_training = False
        print("Skip Training")
    else:
        print("Start Training")
    batch_one = next(iter(train_loader))
    if start_training:
        loss_function = nn.NLLLoss(ignore_index=padding_idx).to(device)
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
            # it insures that the sequnce is not messed up during reshape
            loss = loss_function(log_prediction, out_captions)
            loss.backward()
            print("Loss:", loss.item())
            # step 5. use optimizer to take gradient step
            optimizer.step()

        end = timer()
        print("Overall Learning Time", end - start)
        torch.save(network.state_dict(), model_path)
    # c_vectorizer.max_sequence_length - 1 because the length is one shorter if we only predict...

    begin_word_idx = c_vectorizer.caption_vocab._token_to_idx[BEGIN_WORD]
    starting_token = torch.ones(c_vectorizer.max_sequence_length - 1, dtype=torch.long).to(device) * padding_idx
    starting_token[0] = begin_word_idx
    images, in_captions, out_captions = model.CocoDatasetWrapper.transform_batch_for_training(batch_one, device)

    network.eval()
    #TODO implement the whole sequence prediction using beam search...
    pred = network((images[0].unsqueeze(dim=0),starting_token.unsqueeze(dim=0).unsqueeze(dim=0)))
    first_predicted_idx = pred[0][0].argmax().item()
    predicted_word = c_vectorizer.caption_vocab._idx_to_token[first_predicted_idx]
    print("predicted word:", predicted_word)
    
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

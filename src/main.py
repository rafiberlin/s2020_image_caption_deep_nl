import torch
import torchvision.transforms as transforms
import torch.utils.data
import os
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
# own modules
import model
import preprocessing


def get_hyper_parameters():
    device = get_device()
    parameters = OrderedDict([("lr", [0.01, 0.001]),
                              ("batch_size", [10, 100, 100]),
                              ("shuffle", [True, False]),
                              ("epochs", [10, 100]),
                              ("device", device)
                              ])
    return parameters


def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


def get_dataset_file_args():
    file_args = {"train": {"img": "./data/train2017", "inst": "./data/annotations/instances_train2017.json",
                           "capt": "./data/annotations/captions_train2017.json"},
                 "val": {"img": "./data/val2017", "inst": "./data/annotations/instances_val2017.json",
                         "capt": "./data/annotations/captions_val2017.json"}
                 }
    return file_args


DATASET_FILE_PATHS_CONFIG = "./dataset_file_args.json"
HYPER_PARAMETER_CONFIG = "./hyper_parameters.json"
N_EPOCHS = 120
LEARNING_RATE = 0.01
REPORT_EVERY = 5
EMBEDDING_DIM = 30
HIDDEN_DIM = 20
HIDDEN_DIM_CNN = 100
HIDDEN_DIM_RNN = 100
BATCH_SIZE = 150
N_LAYERS = 1
PADDING_WORD = "<MASK>"

def main():
    file_args = preprocessing.read_json_config(DATASET_FILE_PATHS_CONFIG)
    hyper_parameters = preprocessing.read_json_config(HYPER_PARAMETER_CONFIG)
    cleaned_captions = preprocessing.create_list_of_captions_and_clean(file_args["train"]["annotation_dir"],
                                                                       file_args["train"]["capt"])
    c_vectorizer = model.CaptionVectorizer.from_dataframe(cleaned_captions)
    words = c_vectorizer.caption_vocab._token_to_idx.keys()
    padding_idx = c_vectorizer.caption_vocab._token_to_idx["<MASK>"]
    #embeddings = model.make_embedding_matrix(glove_filepath=file_args["embeddings"],
    #                                         words=words)

    image_dir = file_args["train"]["img"]
    caption_file_path = os.path.join(file_args["train"]["annotation_dir"], file_args["train"]["capt"])
    rgb_stats = preprocessing.read_json_config(file_args["rgb_stats"])
    stats_rounding = hyper_parameters["rounding"]
    rgb_mean = tuple([round(m, stats_rounding) for m in rgb_stats["mean"]])
    rgb_sd = tuple([round(s, stats_rounding) for s in rgb_stats["mean"]])
    # TODO create a testing split, there is only training and val currently...
    coco_train_set = dset.CocoDetection(root=image_dir,
                                        annFile=caption_file_path,
                                        transform=transforms.Compose([preprocessing.CenteringPad(),
                                                                      transforms.Resize((640, 640)),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(rgb_mean, rgb_sd)])
                                        )

    coco_dataset_wrapper = model.CocoDatasetWrapper(coco_train_set, c_vectorizer)
    train_loader = torch.utils.data.DataLoader(coco_dataset_wrapper, hyper_parameters["batch_size"][0])
    batch_one = next(iter(train_loader))
    img, capt = batch_one[0], batch_one[1]



    vocabulary_size = len(c_vectorizer.caption_vocab)

    network = model.LSTMModel(EMBEDDING_DIM, vocabulary_size, HIDDEN_DIM_RNN, HIDDEN_DIM_CNN, padding_idx)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(params=network.parameters(), lr=LEARNING_RATE)
    network.zero_grad()
    # flatten all caption
    f = capt[0]["vectorized_caption"][1]
    expected = capt[0]["vectorized_caption"][1].reshape(-1)
    # flatten all batch and sequences, to make it category comparable
    # for the loss function
    log_prediction = network(batch_one)
    log_prediction = log_prediction.reshape(expected.shape[0], -1)


    # TODO
    loss = loss_function(log_prediction,  expected.reshape(-1))


    from timeit import default_timer as timer

    start = timer()
    # --- training loop ---
    for epoch in range(N_EPOCHS):
        total_loss = 0

        # Generally speaking, it's a good idea to shuffle your
        # datasets once every epoch.

        # WRITE CODE HERE
        # Sort your training set according to word-length,
        # so that similar-length words end up near each other
        # You can use the function get_word_length as your sort key.

        # When I sort the trainset based on word length,
        # the learning get stuck in a local minimum with accuracy at around 16 percent
        # on training data, this is why I am not using it
        # trainset = sorted(trainset, key=get_word_length)

        for i in range(0, len(trainset), BATCH_SIZE):
            minibatchwords = trainset[i:i + BATCH_SIZE]

            # print(minibatchwords)
            model.zero_grad()
            mb_x, mb_y = get_minibatch(minibatchwords, character_map, languages)
            # WRITE CODE HERE
            log_prediction = model(mb_x)
            loss = loss_function(log_prediction, mb_y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print('epoch: %d, loss: %.4f' % ((epoch + 1), total_loss))


    end = timer()
    print("Overall Learning Time", end - start)

if __name__ == '__main__':
    main()


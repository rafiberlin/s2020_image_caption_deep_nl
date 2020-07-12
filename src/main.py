#!/usr/bin/env python3
import torch
import torch.utils.data
import os
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
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
        prep.download_unpack_zip(hparams["img_train_url"], hparams["root"])
        prep.download_unpack_zip(hparams["img_val_url"], hparams["root"])
        prep.download_unpack_zip(hparams["glove_url"], hparams["root"])
        with open("./utils/glove_conv.py") as script_file:
            exec(script_file.read())
    trainset_name = "val"
    #trainset_name = "test"
    valset_name = "val"
    testset_name ="test"
    device = hparams["device"]
    if not torch.cuda.is_available():
        print("Warning, only CPU processing available!")
        device = "cpu"
    else:
        print("CUDA GGP is available", "Number of machines:", torch.cuda.device_count())

    cleaned_captions = prep.create_list_of_captions_and_clean(hparams, trainset_name)
    c_vectorizer = model.CaptionVectorizer.from_dataframe(cleaned_captions)
    padding_idx = None
    if(hparams["use_padding_idx"]):
        padding_idx = c_vectorizer.get_vocab()._token_to_idx[PADDING_WORD]

    embedding = model.create_embedding(hparams, c_vectorizer, padding_idx)

    train_loader = model.CocoDatasetWrapper.create_dataloader(hparams, c_vectorizer, trainset_name)
    #The last parameter is needed, because the images of the testing set ar in the same directory as the images of the training set
    #test_loader = model.CocoDatasetWrapper.create_dataloader(hparams, c_vectorizer, testset_name, "val2017")
    val_loader = model.CocoDatasetWrapper.create_dataloader(hparams, c_vectorizer, valset_name)

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
    network = model.LSTMModel(hparams["hidden_dim_rnn"], hparams["hidden_dim_cnn"], pretrained_embeddings=embedding, cnn_model=hparams["cnn_model"]).to(device)
    #network = model.LSTMModel(EMBEDDING_DIM, vocabulary_size, HIDDEN_DIM_RNN, HIDDEN_DIM_CNN, pretrained_embeddings=embeddings).to(device)
    start_training = True
    if os.path.isfile(model_path) and not args.train:
        network.load_state_dict(torch.load(model_path))
        start_training = False
        print("Skip Training")
    else:
        print("Start Training")
    #Set "break_training_loop_percentage" to 100 in hparams.json to train on everything...
    batch_size = hparams["batch_size"][0]
    break_training_loop_percentage = hparams["break_training_loop_percentage"]
    break_training_loop_idx = max(int(len(train_loader)/batch_size*break_training_loop_percentage/100) - 1, 0)
    #break_val_loop_idx = max(int(len(val_loader)/batch_size*break_training_loop_percentage/100) - 1, 0)
    #break_test_loop_idx = max(int(len(test_loader)/batch_size*break_training_loop_percentage/100) - 1, 0)

    if start_training:
        loss_function = nn.NLLLoss().to(device)
        optimizer = optim.Adam(params=network.parameters(), lr=hparams['lr'])
        start = timer()
        # --- training loop ---
        torch.cuda.empty_cache()
        network.train()
        total_loss = 0
        for epoch in range(hparams["num_epochs"]):
            for idx , current_batch in enumerate(train_loader):
                images, in_captions, out_captions = model.CocoDatasetWrapper.transform_batch_for_training(current_batch, device)
                optimizer.zero_grad()
                # flatten all caption , flatten all batch and sequences, to make its category comparable
                # for the loss function
                out_captions = out_captions.reshape(-1)
                log_prediction = network((images, in_captions)).reshape(out_captions.shape[0], -1)
                # Warning if we are unable to learn, use the contiguous function of the tensor
                # it insures that the sequence is not messed up during reshape
                loss = loss_function(log_prediction, out_captions)
                loss.backward()
                # Use optimizer to take gradient step
                optimizer.step()
                #for dev purposes only
                if idx == break_training_loop_idx:
                    break
            if (epoch+1) % 10 == 0:
                print("Loss:", loss.item(), "Epoch:", epoch+1)
        end = timer()
        print("Overall Learning Time", end - start)
        torch.save(network.state_dict(), model_path)

    model.BleuScorer.perform_whole_evaluation(hparams, train_loader, network, c_vectorizer, break_training_loop_idx)
    #model.BleuScorer.perform_whole_evaluation(test_loader, network, c_vectorizer, break_test_loop_idx, hparams["print_prediction"])
    #model.BleuScorer.perform_whole_evaluation(val_loader, network, c_vectorizer, break_val_loop_idx, hparams["print_prediction"])

if __name__ == '__main__':
    main()

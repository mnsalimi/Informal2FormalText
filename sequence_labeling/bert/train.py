import os
import pickle
import argparse
import matplotlib.pyplot as plt
import torch
from utils import Meter
from TorchCRF import CRF
from transformers import *
from load import load_data
from utils import print_info
from model import LangModelWithDense
# from tqdm import tqdm


def train_model(model,
                train_loader, dev_loader,
                optimizer, criterion,
                num_classes, target_classes,
                label_encoder,
                device):

    # create to Meter's classes to track the performance of the model during training and evaluating
    train_meter = Meter(target_classes)
    dev_meter = Meter(target_classes)

    best_f1 = -1

    # epoch loop
    best_macro_f1 = []
    train_losses = []
    train_f1s = []
    dev_losses = []
    dev_f1s = []
    for epoch in range(args.epochs):
        # train_tqdm = tqdm(train_loader)
        # dev_tqdm = tqdm(dev_loader)

        model.train()

        # train loop
        # for i, (train_x, train_y, mask, crf_mask) in enumerate(train_tqdm):
        for i, (train_x, train_y, mask, crf_mask) in enumerate(train_loader):
            # get the logits and update the gradients
            optimizer.zero_grad()

            logits = model.forward(train_x, mask)

            if args.no_crf:
                loss = criterion(logits.reshape(-1, num_classes).to(device), train_y.reshape(-1).to(device))
            else:
                loss = - criterion(logits.to(device), train_y, reduction="token_mean", mask=crf_mask)

            loss.backward()
            optimizer.step()

            # get the current metrics (average over all the train)
            loss, _, _, micro_f1, _, _, macro_f1 = train_meter.update_params(loss.item(), logits, train_y)
        train_losses.append(loss)
        train_f1s.append(macro_f1)
            # print the metrics
            # train_tqdm.set_description("Epoch: {}/{}, Train Loss: {:.4f}, Train Micro F1: {:.4f}, Train Macro F1: {:.4f}".
            #                            format(epoch + 1, args.epochs, loss, micro_f1, macro_f1))
        print("Epoch: {}/{}, Train Loss: {:.4f}, Train Macro F1: {:.4f}".
                                    format(epoch + 1, args.epochs, loss, macro_f1))
            # train_tqdm.refresh()

        # reset the metrics to 0
        train_meter.reset()

        model.eval()

        # evaluation loop -> mostly same as the training loop, but without updating the parameters
        # for i, (dev_x, dev_y, mask, crf_mask) in enumerate(dev_tqdm):
        for i, (dev_x, dev_y, mask, crf_mask) in enumerate(dev_loader):
            logits = model.forward(dev_x, mask)

            if args.no_crf:
                loss = criterion(logits.reshape(-1, num_classes).to(device), dev_y.reshape(-1).to(device))
            else:
                loss = - criterion(logits.to(device), dev_y, reduction="token_mean", mask=crf_mask)

            loss, _, _, micro_f1, _, _, macro_f1 = dev_meter.update_params(loss.item(), logits, dev_y)

            # dev_tqdm.set_description("Dev Loss: {:.4f}, Dev Micro F1: {:.4f}, Dev Macro F1: {:.4f}\n".
            #                          format(loss, micro_f1, macro_f1))
        dev_losses.append(loss)
        dev_f1s.append(macro_f1)
        print("Dev Loss: {:.4f}, Dev Macro F1: {:.4f}".
                                    format(loss, macro_f1))
            # dev_tqdm.refresh()

        dev_meter.reset()

        # if the current macro F1 score is the best one -> save the model
        if macro_f1 > best_f1:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            # print("Macro F1 score improved from {:.4f} -> {:.4f}. Saving model...".format(best_f1, macro_f1))

            best_f1 = macro_f1
            best_f1_epoch = epoch
            torch.save(model, os.path.join(args.save_path, "model.pt"))
            with open(os.path.join(args.save_path, "label_encoder.pk"), "wb") as file:
                pickle.dump(label_encoder, file)
        print("The BEST F1 on DEV: {} at epoch {}\n".format(best_f1, best_f1_epoch))
    plot_graph(train_losses, dev_losses, args)

def plot_graph(Y1, Y2, args):
  
    # line 1 points
    # plotting the line 1 points 
    X1 = [i+1 for i in range(len(Y1))]
    plt.plot(X1, Y1, label = "train loss")
    
    # line 2 points
    # plotting the line 2 points 
    plt.plot(X1, Y2, label = "test loss")
    
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    config = 'epochs={}, maxlen={}, batch_size={}'.format(args.epochs, args.max_len, args.batch_size)
    plt.title(config)
    plt.legend()
    plt.savefig('config.png')

def main():
    device = torch.device(args.device)
    # Loading the train and dev data and save them in a loader + the encoder of the classes
    train_loader, dev_loader, label_encoder = load_data(args.train_path,
                                                        args.dev_path,
                                                        args.batch_size,
                                                        args.tokens_column, args.predict_column,
                                                        args.lang_model_name,
                                                        args.max_len,
                                                        args.separator,
                                                        args.pad_label,
                                                        args.null_label,
                                                        device)

    # select the desired language model and get the embeddings size
    lang_model = AutoModel.from_pretrained(args.lang_model_name)
    input_size = 768 if "base" in args.lang_model_name else 1024

    # create the model, the optimizer (weights are set to 0 for <pad> and <X>) and the loss function
    model = LangModelWithDense(lang_model, input_size, len(label_encoder.classes_), args.fine_tune).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    if args.no_crf:
        weights = torch.tensor([1 if label != args.pad_label and label != args.null_label else 0 for label in label_encoder.classes_], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = CRF(len(label_encoder.classes_),).to(device)

    # remove the null_label (X), the pad label (<pad>) and the (O)-for NER only from the evaluated targets during training
    classes = label_encoder.classes_.tolist()
    try:
        classes.remove(args.null_label)
    except:
        pass
    try:
        classes.remove(args.pad_label)
    except:
        pass
    # classes.remove("O")
    target_classes = [label_encoder.transform([clss])[0] for clss in classes]

    print_info(target_classes, label_encoder, args.lang_model_name, args.fine_tune, device)

    # start training
    train_model(model,
                train_loader, dev_loader,
                optimizer, criterion,
                len(label_encoder.classes_), target_classes,
                label_encoder,
                device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str, help="Path to the training file")
    parser.add_argument("dev_path", type=str, help="Path to the dev file")
    parser.add_argument("tokens_column", type=int , help="The column of the tokens.")
    parser.add_argument("predict_column", type=int, help="The column that must be predicted")
    parser.add_argument("lang_model_name", type=str, help="Language model name of HuggingFace's implementation.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--save_path", type=str, default="models", help="Where to save the model/")
    parser.add_argument("--fine_tune", action="store_true", help="Use this to fine-tune the language model's weights.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of the files.")
    parser.add_argument("--separator", type=str, default="\t", help="Separator of the tokens in the train/dev files.")
    parser.add_argument("--pad_label", type=str, default="<pad>", help="The pad token.")
    parser.add_argument("--null_label", type=str, default="<X>", help="The null token.")
    parser.add_argument("--no_crf", action='store_true', help="Use this to remove the CRF on top of the language model.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to train on.")

    args = parser.parse_args()

    main()
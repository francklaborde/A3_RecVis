import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        # nargs="+",
        metavar="M",
        help="SGD momentum (default: 0.5)/ Adam (beta1, beta2) (default: (0.9, 0.999))",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--show_structure",
        type=bool,
        default=False,
        help="Display the structure of the model selected"
    )
    parser.add_argument(
        "--show_loss",
        type=bool,
        default=False,
        help="Display the training loss and the validation loss on graphs at the end of the training"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer to use"
    )
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item()/len(train_loader.dataset),
                )
            )
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    return loss.data.item()/len(train_loader.dataset)


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return validation_loss

# def get_optimizer(optimizer_name: str,
#                   model: nn.Module,
#                   lr: float,
#                   args:argparse.ArgumentParser
# ) -> torch.optim.Optimizer:

#     momentum = args.momentum
#     if optimizer_name == "SGD":
#         if len(momentum) == 1:
#             print("SGD with momentum: ", momentum[0])
#             return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum[0])
#         else:
#             raise ValueError("SGD needs only one momentum value")
#     elif optimizer_name == "Adam":
#         if len(momentum) == 1:
#             #Value by default for the momentum
#             print("Adam with default momentum: ", (0.9, 0.999))
#             return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
#         elif len(momentum) == 2 or len(momentum) == 1:
#             print("Adam with momentum: ", (momentum[0], momentum[1]))
#             return optim.Adam(model.parameters(), lr=lr, betas=(momentum[0], momentum[1]))
#         else:
#             raise ValueError("Adam needs two momentum values")   

 
def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms = ModelFactory(args.model_name).get_all()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.show_structure:
        print(model)
    if args.show_loss:
        train_losses = []
        val_losses = []
    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # training loop
        train_loss = train(model, optimizer, train_loader, use_cuda, epoch, args)
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        model_file = args.experiment + "/model_" + args.model_name + "_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        if args.show_loss:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
    if args.show_loss:
        # Création de la figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Sous-graphe 1 : Pertes d'entraînement
        ax1.plot(train_losses, label="Training Loss", marker='o', linestyle='-', color='blue')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(visible=True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Sous-graphe 2 : Pertes de validation
        ax2.plot(val_losses, label="Validation Loss", marker='s', linestyle='--', color='orange')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Loss")
        ax2.grid(visible=True, linestyle='--', alpha=0.7)
        ax2.legend()

        # Ajustement de l'espacement entre les sous-graphiques
        plt.tight_layout()

        # Sauvegarde de la figure
        output_filename = f"loss_plots_{args.model_name}.png"
        plt.savefig(output_filename)
        print(f"Figure saved as: {output_filename}")

from torchvision.models import ResNet50_Weights

if __name__ == "__main__":
    main()

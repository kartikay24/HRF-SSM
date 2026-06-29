import torch.nn
import torchvision
import tools
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import math
import random
import os

import sys
sys.path.append("..")
import snn

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Train SMNIST with SimpleHarmonicRNN")

# Add arguments for each parameter
parser.add_argument("--omega_a", type=float, default=10.0, help="Lower bound for omega uniform distribution")
parser.add_argument("--omega_b", type=float, default=50.0, help="Upper bound for omega uniform distribution")
parser.add_argument("--b_offset_a", type=float, default=1.0, help="Lower bound for b_offset uniform distribution")
parser.add_argument("--b_offset_b", type=float, default=6.0, help="Upper bound for b_offset uniform distribution")
parser.add_argument("--out_adaptive_tau_mem_mean", type=float, default=0.0, help="Mean for LI alpha normal distribution")
parser.add_argument("--out_adaptive_tau_mem_std", type=float, default=0.1, help="Std for LI alpha normal distribution")
parser.add_argument("--mask_prob", type=float, default=0.0, help="Fraction of elements in hidden.linear.weight to be zero")
parser.add_argument("--scheme", type=str, default="imex2", choices=["euler forward", "euler backward", "imex", "imex2"], help="Integration scheme")
parser.add_argument("--data", type=str, default="smnist", choices=["smnist","shd"], help="Choose Classification task")
parser.add_argument("--permuted", action="store_true", help="Apply fixed random permutation to MNIST data")

# Parse arguments
args = parser.parse_args()

################################################################
# General settings
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device == "cuda":
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = 0

print(device)

################################################################
# Data loading and preparation, logging
################################################################

rand_num = random.randint(1, 10000)
start_time = datetime.now().strftime("%m-%d_%H-%M-%S")

if args.data == "smnist":
    PERMUTED = args.permuted
    label_last = False

    sequence_length = 28 * 28
    input_size = 1
    num_classes = 10
    batch_size = 256  # (256 from Yin et al. 2021)

    # validation and test batch size can be chosen higher
    # (depending on VRAM capacity)
    val_batch_size = 256
    test_batch_size = 256


    # PSMNIST fixed random permutation
    if PERMUTED:
        permuted_idx = torch.randperm(sequence_length)
        torch.save(permuted_idx, './models/{}'.format(start_time) + '_' + str(rand_num) + '_permuted_idx.pt')
    else:
        permuted_idx = torch.arange(sequence_length)


    train_dataset = torchvision.datasets.MNIST(
        root=r"smnist/MNIST_data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    total_dataset_size = len(train_dataset)

    # we use 5% - 10% of the training data for validation
    val_dataset_size = int(total_dataset_size * 0.1)
    train_dataset_size = total_dataset_size - val_dataset_size

    train_dataset, val_dataset = random_split(
        train_dataset, [train_dataset_size, val_dataset_size]
    )

    test_dataset = torchvision.datasets.MNIST(
        root=r"smnist/MNIST_data",
        train=False,
        transform=torchvision.transforms.ToTensor()
    )

if args.data == "shd":
    # TRAIN DATASET #
    whole_train_dataset = tools.shd_to_dataset('./SHD/SHD_data/trainX_4ms.npy', './SHD/SHD_data/trainY_4ms.npy')


    # 8156 sequences in whole training dataset
    total_train_dataset_size = len(whole_train_dataset)

    # 10 % of training data used for validation -> 815
    val_dataset_size = int(total_train_dataset_size * 0.1)

    # 7341 sequences used for training
    train_dataset_size = total_train_dataset_size - val_dataset_size

    # split whole train dataset randomly
    train_dataset, val_dataset = random_split(
        dataset=whole_train_dataset,
        lengths=[train_dataset_size, val_dataset_size]
    )

    # TEST DATASET #
    test_dataset = tools.shd_to_dataset('./SHD/SHD_data/testX_4ms.npy', './SHD/SHD_data/testY_4ms.npy')

    # 2264 sequences in test dataset
    test_dataset_size = len(test_dataset)

    label_last = False
    sequence_length = 250
    input_size = 700
    hidden_size = 128

    num_classes = 20
    batch_size = 32

    # validation and test batch size can be chosen higher
    # (depending on VRAM capacity)
    val_batch_size = 256
    test_batch_size = 256

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
)

test_dataset_size = len(test_dataset)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
)

def smnist_transform_input_batch(
        tensor: torch.Tensor,
        sequence_length_: int,
        batch_size_: int,
        input_size_: int,
        permuted_idx_: torch.Tensor
):
    tensor = tensor.to(device=device).view(batch_size_, sequence_length_, input_size_)
    tensor = tensor.permute(1, 0, 2)
    tensor = tensor[permuted_idx_, :, :]
    return tensor

################################################################
# Model helpers and model setup
################################################################

hidden_size = 256

criterion = torch.nn.NLLLoss()


# Assign parsed arguments to variables
omega_a = args.omega_a
omega_b = args.omega_b
b_offset_a = args.b_offset_a
b_offset_b = args.b_offset_b
out_adaptive_tau_mem_mean = args.out_adaptive_tau_mem_mean
out_adaptive_tau_mem_std = args.out_adaptive_tau_mem_std
mask_prob = args.mask_prob
scheme = args.scheme

model_params = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": num_classes,  # Assuming `num_classes` is the output size
    "adaptive_omega_a": omega_a,
    "adaptive_omega_b": omega_b,
    "adaptive_b_offset_a": b_offset_a,
    "adaptive_b_offset_b": b_offset_b,
    "out_adaptive_tau_mem_mean": out_adaptive_tau_mem_mean,
    "out_adaptive_tau_mem_std": out_adaptive_tau_mem_std,
    "label_last": label_last,
    "hidden_bias": False,  # Optional, default is False
    "output_bias": False,   # Optional, default is False
    "scheme": scheme  # Optional, default is 'imex'
}

# import pdb; pdb.set_trace()
model = snn.models.SimpleHarmonicRNN(**model_params).to(device)

# TORCH SCRIPT #
model = torch.jit.script(model)
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(num_params)
# 68874

################################################################
# Setup experiment (optimizer etc.)
################################################################

optimizer_lr = 0.1
gradient_clip_value = 1.

optimizer = torch.optim.RAdam(model.parameters(), lr=optimizer_lr)

# Number of iterations per epoch
total_steps = len(train_loader)
epochs_num = 300
padding = 0

# learning rate scheduling
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs_num)

if args.data == "smnist":
    # [logging]: CHECK that omega and b_offset inits are correctly distributed!
    opt_str = "{}_RAdam({}),NLL,script-bw,LinLR,LL({}),data(SMNIST),PERMUTED({})"\
        .format(rand_num, optimizer_lr, label_last, PERMUTED)
    net_str = "1,{},10,bs={},ep={},scheme={}".format(hidden_size, batch_size, epochs_num, scheme)
    unit_str = "HRF_omega{}_{}b{}_{},LI{}_{}"\
        .format(omega_a, omega_b, b_offset_a, b_offset_b, out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std) #

    comment = opt_str + "," + net_str + "," + unit_str

    writer = SummaryWriter(comment=comment)

if args.data == "shd":
    # [logging]: CHECK that omega and b_offset inits are correctly distributed!
    opt_str = "{}_RAdam({}),NLL,script-bw,LinLR,LL({}),data(SHD)"\
        .format(rand_num, optimizer_lr, label_last)
    net_str = "1,{},10,bs={},ep={},scheme={}".format(hidden_size, batch_size, epochs_num, scheme)
    unit_str = "HRF_omega{}_{}b{}_{},LI{}_{}"\
        .format(omega_a, omega_b, b_offset_a, b_offset_b, out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std) #

    comment = opt_str + "," + net_str + "," + unit_str

    writer = SummaryWriter(comment=comment)

save_path = "models/{}_".format(start_time) + comment + ".pt"
save_init_path = "models/{}_init_".format(start_time) + comment + ".pt"

print(start_time, comment)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# save initial parameters for analysis
torch.save({'model_state_dict': model.state_dict()}, save_init_path)

# print(model.state_dict())
print_every = 150
################################################################
# Training loop
################################################################

iteration = 0
min_val_loss = float("inf")
loss_value = 1.
end_training = False

run_time = tools.PerformanceCounter()
tools.PerformanceCounter.reset(run_time)

for epoch in range(epochs_num + 1):

    # check initial performance without training (for plotting purposes)
    # Go into eval mode
    model.eval()

    with torch.no_grad():

        val_loss = 0
        val_correct = 0

        # Perform validation
        for i, (inputs, targets) in enumerate(val_loader):
            current_batch_size = len(inputs)

            if args.data == "smnist":
                # Reshape inputs in [sequence_length, batch_size, data_size].
                input = smnist_transform_input_batch(
                    tensor=inputs.to(device=device),
                    sequence_length_=sequence_length,
                    batch_size_=current_batch_size,
                    input_size_=input_size,
                    permuted_idx_=permuted_idx
                )
            if args.data == "shd":                
                # Reshape inputs in [sequence_length, batch_size, data_size].
                input = inputs.permute(1, 0, 2).to(device)

            target = targets.to(device=device)

            outputs, _, _ = model(input)

            # Apply loss sequentially against single pattern.
            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            # for Label Last
            if label_last:
                val_loss_value = loss.item()
            else:
                val_loss_value = loss.item() / sequence_length

            val_loss += val_loss_value

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)
            val_correct += batch_correct

        val_loss /= len(val_loader)  # val_dataset_size
        val_accuracy = (val_correct / val_dataset_size) * 100.0

        # Log current val loss and accuracy
        writer.add_scalar(
            "Loss/val",
            val_loss,
            epoch
        )
        writer.add_scalar(
            "Accuracy/val",
            val_accuracy,
            epoch
        )

        # Persist current best model.
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            best_model_state_dict = model.state_dict()
            # TODO save checkpoint of the training including model.state_dict() and optimizer.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, save_path)

        test_loss = 0
        test_correct = 0
        test_total_spikes = 0.

        # Perform Inference
        for i, (inputs, targets) in enumerate(test_loader):
            current_batch_size = len(inputs)

            if args.data == "smnist":
                # Reshape inputs in [sequence_length, batch_size, data_size].
                input = smnist_transform_input_batch(
                    tensor=inputs.to(device=device),
                    sequence_length_=sequence_length,
                    batch_size_=current_batch_size,
                    input_size_=input_size,
                    permuted_idx_=permuted_idx
                )
            if args.data == "shd":                
                # Reshape inputs in [sequence_length, batch_size, data_size].
                input = inputs.permute(1, 0, 2).to(device)

            # Reshape targets (for MNIST it's a single pattern).
            target = targets.to(device=device)

            outputs, _, num_spikes1 = model(input)

            # accumulate total spikes
            test_total_spikes += num_spikes1 #.item()

            # Apply loss sequentially against single pattern.
            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            # for Label Last
            if label_last:
                test_loss_value = loss.item()
            else:
                test_loss_value = loss.item() / sequence_length

            test_loss += test_loss_value

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)
            test_correct += batch_correct

        test_loss /= len(test_loader)  # test_dataset_size
        test_accuracy = (test_correct / test_dataset_size) * 100.0
        test_sop = test_total_spikes / test_dataset_size

        # Log current test loss and accuracy
        writer.add_scalar(
            "Loss/test",
            test_loss,
            epoch
        )
        writer.add_scalar(
            "Accuracy/test",
            test_accuracy,
            epoch
        )
        writer.add_scalar(
            "SOP/test",
            test_sop,
            epoch
        )
        print(
            "Epoch [{:4d}/{:4d}]  |  Summary  |  Loss/val: {:.6f}, Accuracy/val: {:.4f}%  |  Loss/test: {:.6f}, "
            "Accuracy/test: {:.4f} | SOP: {:.4f}".format(
                epoch, epochs_num, val_loss, val_accuracy, test_loss, test_accuracy, test_sop), flush=True
        )

    # Update logging outputs
    writer.flush()

    if epoch < epochs_num:

        # Go into train mode.
        model.train()

        train_correct = 0
        print_train_loss = 0
        print_correct = 0
        print_total = 0

        # Perform training epoch (iterate over all mini batches in training set).
        for i, (inputs, targets) in enumerate(train_loader):
            current_batch_size = len(inputs)
        
            if args.data == "smnist":
                # Reshape inputs in [sequence_length, batch_size, data_size].
                input = smnist_transform_input_batch(
                    tensor=inputs.to(device=device),
                    sequence_length_=sequence_length,
                    batch_size_=current_batch_size,
                    input_size_=input_size,
                    permuted_idx_=permuted_idx
                )
            if args.data == "shd":                
                # Reshape inputs in [sequence_length, batch_size, data_size].
                input = inputs.permute(1, 0, 2).to(device)

            # Reshape targets (for MNIST it's a single pattern).
            target = targets.to(device=device)

            # Clear previous gradients
            optimizer.zero_grad()

            outputs, _, _ = model(input)

            # Apply loss sequentially against single pattern.
            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            # for Label Last
            if label_last:
                loss_value = loss.item()
            else:
                loss_value = loss.item() / sequence_length

            # calculate gradient
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

            # Perform learning step
            optimizer.step()

            if math.isnan(loss_value):
                end_training = True
                break

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)

            # # Log current loss and accuracy
            writer.add_scalar(
                "Loss/train",
                loss_value,
                iteration
            )
            writer.add_scalar(
                "Accuracy/train",
                (batch_correct / current_batch_size) * 100.0,
                iteration
            )

            print_train_loss += loss_value
            print_total += current_batch_size
            print_correct += batch_correct

            # Print current training loss/acc at every 50th iteration
            if i % print_every == (print_every - 1):
                print_acc = (print_correct / print_total) * 100.0

                print("Epoch [{:4d}/{:4d}]  |  Step [{:4d}/{:4d}]  |  Loss/train: {:.6f}, Accuracy/train: {:8.4f}".format(
                    epoch + 1, epochs_num, i + 1, total_steps, print_train_loss / print_every, print_acc), flush=True
                )

                print_correct = 0
                print_total = 0
                print_train_loss = 0

            iteration += 1

        # next in lr scheduler
        scheduler.step()

        # Update logging outputs
        writer.flush()


    if end_training:
        break

writer.close()
print("Minimum val loss: {:.6f} at epoch: {}".format(min_val_loss, min_val_epoch))
print(tools.PerformanceCounter.time(run_time) / 3600, "hr")

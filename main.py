
import sys

import torch
import torch.nn.functional as F
import torchvision as torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from general_utils import get_path_for_graph
from main_utils import *
from model_utils import *


def train(name,
          dataset_name, should_log_metadata,
          batch_size, should_shuffle, num_workers,
          criterion_name,
          optimizer_name, learning_rate, weight_decay, momentum,
          num_epochs, print_every):

    print('Training ' + str(name))

    model = create_model()

    data_loader = create_data_loader(dataset_name, should_log_metadata, batch_size, should_shuffle, num_workers)

    criterion = create_criterion(criterion_name)

    optimizer = create_optimizer(model, optimizer_name, weight_decay, momentum, learning_rate)

    _train_internal(model, data_loader, criterion, optimizer, num_epochs, print_every, name)


def create_data_loader(dataset_name, should_log_metadata, batch_size, should_shuffle, num_workers):
    if dataset_name in ('train.pickle', 'dev.pickle'):
        dataset = load_pickle_dataset(dataset_name=dataset_name, should_log_metadata=should_log_metadata)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset = torchvision.datasets.ImageFolder(get_path_to_image_folder_root(dataset_name),
                                                   transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle, num_workers=num_workers)


def create_criterion(criterion_name):
    # todo: figure out why didn't work (for next exercise, cel is da bast for ex1)
    if criterion_name == 'cross_entropy_loss':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'nll_loss':
        raise Exception("Error: nll_loss doesn't work")
        # return nn.NLLLoss()
    elif criterion_name == 'MSE_loss':
        raise Exception("Error: MSE_loss doesn't work")
        # return nn.MSELoss()
    elif criterion_name == 'hinge_loss':
        raise Exception("Error: hinge_loss doesn't work")
        # return nn.HingeEmbeddingLoss()
    else:
        raise Exception('Error: unsupported criterion_name:', criterion_name)


def create_optimizer(model, optimizer_name, weight_decay, momentum, learning_rate):
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise Exception('Error: unsupported optimizer_name:', optimizer_name)


def _train_internal(model, data_loader, criterion, optimizer, num_epochs, print_every, name, should_plot=True):
    print('Training is starting now.')

    total_losses = list()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_losses = list()

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == (print_every - 1):  # print every $print_every mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0

        total_losses.append(np.mean(epoch_losses))

    print('Training completed.')
    save_checkpoint(model, name)

    if should_plot:
        plt.title("Training loss by epoch - Round " + str(name))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(total_losses)
        plt.yticks(np.arange(0.25, 0.71, step=0.05))
        plt.savefig(get_path_for_graph(name))
        plt.show()

    validate()
    test()


def validate():
    print('Validating.')

    _test_internal('current_weights', 'dev.pickle')


def test():
    print('Testing.')

    _test_internal('current_weights', 'organized_data/')


def validate_pre_trained():
    print('Validating pre-trained model.')

    _test_internal('pre_trained', 'dev.pickle')


def test_pre_trained():
    print('Testing pre-trained model.')

    _test_internal('pre_trained', 'organized_data/')


def _test_internal(checkpoint_name, path_to_dataset):
    model = create_model(checkpoint_name=checkpoint_name, is_in_eval_mode=True)

    data_loader = create_data_loader(path_to_dataset, True, 4, False, 2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # todo: len(data_loader) gives (num_images / batch_size), not num_images
    print('Accuracy of the network on the ' + str(len(data_loader)) + ' test images: %d %%' % (100 * correct / total))


def findAdversarialExample(attacker_name='FGSM',
                           use_cuda=False, checkpoint_name='pre_trained', path_to_dataset='dev.pickle', epsilons=None):
    if epsilons is None:
        epsilons = [0, .05, .1, .15, .2, .25, .3]

    model = create_model(checkpoint_name=checkpoint_name, is_in_eval_mode=True)
    data_loader = create_data_loader(path_to_dataset, False, 1, True, 2)
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    attacker = create_attacker(attacker_name)

    accuracies = []
    examples = []
    for epsilon in epsilons:
        acc, ex = _findAdversarialExampleInternal(model, data_loader, device, attacker, epsilon)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, max(epsilons) + 0.05, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


def create_attacker(attacker_name):
    if attacker_name == 'FGSM':
        return fgsm_attack
    else:
        raise Exception('Error: unsupported algorithm_name:', attacker_name)


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def _findAdversarialExampleInternal(model, data_loader, device, attacker, epsilon):
    correct = 0
    adv_examples = []

    for data, target in data_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)

        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        perturbed_data = attacker(data, epsilon, data.grad.data)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(data_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def main():
    if '-train' in sys.argv:
        train(name='37',  # todo make auto increasing!!!
              dataset_name='organized_data/', should_log_metadata=True,
              batch_size=25, should_shuffle=True, num_workers=2,
              criterion_name='cross_entropy_loss',
              optimizer_name='Adam', learning_rate=0.0009, weight_decay=0.1, momentum=0.85,
              num_epochs=10, print_every=50)
    elif '-validate' in sys.argv:
        validate()
    elif '-test' in sys.argv:
        test()
    elif '-validate_pre_trained' in sys.argv:
        validate_pre_trained()
    elif '-test_pre_trained' in sys.argv:
        test_pre_trained()
    elif '-adversarial_example' in sys.argv:
        findAdversarialExample()
    else:
        raise Exception('Error: unsupported sys arguments:', sys.argv)


if __name__ == '__main__':
    main()

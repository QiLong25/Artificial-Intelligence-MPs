import torch
import torch.nn as nn
import numpy as np

def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 5),
    )

def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    return nn.CrossEntropyLoss()

class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################

        self.sequence = nn.Sequential(
            nn.Unflatten(1, (3, 31, 31)),
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),
            nn.Conv2d(in_channels=5, out_channels=7, kernel_size=3, stride=1),
            # nn.Sigmoid(),
            # nn.MaxPool2d(kernel_size=2, ceil_mode=False),
            nn.Flatten(),
            nn.Linear(1008, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 5),
            nn.Sigmoid(),
        )

        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        # X_train = x.numpy()
        # X_mean = np.mean(X_train, axis=0)
        # X_use = X_train - X_mean
        # X_use_max = np.max(np.absolute(X_use), axis=0)
        # X_use = X_use / X_use_max
        # X_use = torch.Tensor(X_use)

        y = self.sequence(x)
        return y

        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = NeuralNet()

    loss_fn = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.03, lr_decay=0.001, weight_decay=0.000001, initial_accumulator_value=0)

    for epoch in range(epochs):
        model.train()

        ## make features and labels
        for features, labels in train_dataloader:
            ## compute loss
            y_pred = model(features)
            loss = loss_fn(y_pred, labels)

            ## gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ################## Your Code Ends here ##################

    return model

"""
Exercise 2: NN als Klasse
"""

import torch
from torch.autograd import Variable
import numpy as np
import archive as A

xy = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

print(x_data.data.shape)
print(y_data.data.shape)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Hidden Layers
        self.layer_1 = torch.nn.Linear(8, 6)
        self.layer_2 = torch.nn.Linear(6, 4)
        self.layer_3 = torch.nn.Linear(4, 1)

        # Aktivierungsfunktion
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.layer_1(x))
        out2 = self.sigmoid(self.layer_2(out1))
        y_pred = self.sigmoid(self.layer_3(out2))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(size_average=True)             # Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)     # Stochastic Gradient Descent

# Training loop
plot = A.NN_Loss_Plot(title='Tools für Coli:  Neural Networks / Exercise 2 (NN als Klasse)', skip=0.1)
for epoch in range(10000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])
    plot.extend_line(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training Complete\n')

A.wait()



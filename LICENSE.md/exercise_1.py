"""
Exercise 1: Gradient Descent
"""

import torch
from torch.autograd import Variable
from time import sleep
import archive as A

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

# Matrix mit nur einem Gewicht mit zufälligem Wert (hier 1)
w = Variable(torch.Tensor([1.0]),  requires_grad=True)

# Wieviele Trainingsdurchläufe?
number_of_epochs = 20

# Learning Rate: In wie großen Schritten soll der Fehler korrigiert werden?
learning_rate = 0.01

# Forward pass
def f(x):
    return x * w

# Funktion für die loss-Berechnung
def loss(x, y_ist):
    y_sol = f(x)
    y_difference = (y_sol - y_ist)
    return y_difference * y_difference

print('\nOutput vor dem Training: f({}) = {:.2f}\n'.format(4, f(4).data[0]))

# Training loop
plot = A.NN_Function_Plot(x_data, y_data, title='Tools für Coli:  Neural Networks / Exercise 1 (Gradient Descent)', skip=1)
for epoch in range(number_of_epochs):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print('\tGrad: {:.2f} {:.2f} {:.2f}'.format(x_val, y_val, w.grad.data[0]))

        # Verbesserung der Werte der Gewichte
        w.data = w.data - learning_rate * w.grad.data

        # Gradienten -> 0, nachdem die Gewichte angepasst wurden
        w.grad.data.zero_()

    plot.add_new_line(x_data, [f(x).data[0] for x in x_data])
    print('Loss nach Trainings Durchlauf Nr. {}: {:.2f}\n'.format(epoch, l.data[0]))
    sleep(0.05)

print('Output nach dem Training: f({}) = {:.2f}\n'.format(4, f(4).data[0]))


A.wait()



'''

Copyright (c) 2018 colipy (https://github.com/colipy)

'''

import matplotlib.pyplot as plt

class NN_Function_Plot:
    def __init__(self, original_x_values, original_y_values, title, skip):
        plt.ion()
        plt.title(title)
        self.fist = True
        self.skip_step = 1 if skip == 0 else skip
        self.skip_count = 0
        self.color_of_new_line = (1, 0, 0, 1)
        self.color_of_old_line = (0, 0, 0, 0.2)
        self.color_of_original_line = (0, 1, 0, 1)
        self.current_line = plt.plot(original_x_values, original_y_values, lw=4, color=self.color_of_original_line)[0]

    def add_new_line(self, x_vals, y_vals):
        # Skip to avoid too much density in the plot
        self.skip_count += self.skip_step
        if self.skip_count >= 1:
            self.skip_count = 0
        else:
            return None

        # Keep the first line as is
        if self.fist:
            self.fist = False
        else:
            self.current_line.set_color(self.color_of_old_line)

        # Draw the new line
        self.current_line = plt.plot(x_vals, y_vals, lw=1, color=self.color_of_new_line)[0]

        # Prevent the from UI freezing
        plt.pause(0.001)


class NN_Loss_Plot:
    def __init__(self, title, skip):
        plt.ion()
        self.x_values = []
        self.y_values = []
        self.y_axis_max = 1
        self.skip_step = 1 if skip == 0 else skip
        self.skip_count = 0
        self.fig, self.ax = plt.subplots()
        self.current_line = self.ax.plot([1, 2, 3], [0, 1, 0], lw=1, color=(0, 1, 0.7, 1))[0]
        plt.title(title)

    def extend_line(self, epoch, loss):
        # Update data
        self.x_values.append(epoch)
        self.y_values.append(loss)
        self.y_axis_max = max([self.y_axis_max, loss])

        # Skip to avoid too much density in the plot
        self.skip_count += self.skip_step
        if self.skip_count >= 1:
            self.skip_count = 0
        else:
            return None

        # Adjust axis
        plt.axis([0, 1 + self.x_values[-1] * 1.05, 0, self.y_axis_max])

        # Draw the new line
        self.current_line.set_xdata(self.x_values)
        self.current_line.set_ydata(self.y_values)

        # Prevent the from UI freezing
        plt.pause(0.001)


def wait():
    while True:
        plt.pause(0.1)


def create_training_file(list_of_input_files, output_file):

    print('create_training_text:', list_of_input_files, '=>', output_file)

    text_list = []
    for input_file in list_of_input_files:
        with open(input_file, 'r') as f:
            text_list.append(f.read())

    with open(output_file, 'w') as f:
        f.write('\n'.join(text_list))
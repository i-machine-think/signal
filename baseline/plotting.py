import matplotlib.pyplot as plt
import csv
import os

def plot_data(filename_data, args):
    iterations = []
    losses = []
    hinge_losses = []
    rl_losses = []
    entropies = []
    accuracies = []

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace("/baseline", "")
    full_filename_data = os.path.join(dir_path, filename_data)

    filename_plot = filename_data.replace("output_data/", "plots/").replace(".csv", ".png")
    full_filename_plot = os.path.join(dir_path, filename_plot)

    with open(full_filename_data,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        index = 0
        for row in plots:
            if index > 0:
                iterations.append(float(row[0]))
                losses.append(float(row[1]))
                if args.rl:
                    hinge_losses.append(float(row[2]))
                    rl_losses.append(float(row[3]))
                    entropies.append(float(row[4]))
                accuracies.append(float(row[5]))
            index +=1

    # plotting taken from https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots()

    # all plots belonging to the first axis
    color = 'tab:red'
    ax1.set_xlabel('Iteration')

    if not args.rl:
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(iterations, losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        ax1.set_ylabel('Loss/Entropy scale', color=color)
        ax1.plot(iterations, losses, label='full loss')
        ax1.plot(iterations, hinge_losses, label='hinge loss')
        ax1.plot(iterations, rl_losses, label='rl loss')
        ax1.plot(iterations, entropies, label='entropy')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)



    # all plots belonging to the second axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1

    ax2.plot(iterations, accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # saving of figure
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(full_filename_plot)

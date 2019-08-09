import matplotlib.pyplot as plt
import csv
import os
import argparse
import pandas as pd


def plot_data(csv_path, used_rl):
    filename, _ = os.path.splitext(csv_path)
    plot_path = filename + '_plot.png'

    df = pd.read_csv(csv_path)

    # plotting taken from https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots()

    # all plots belonging to the first axis
    colors = ["tab:red", "tab:orange", "tab:brown", "tab:pink"]
    ax1.set_xlabel("Iteration")

    if not used_rl:
        ax1.set_ylabel("Loss", color=colors[0])
        ax1.plot(df['iteration'], df['loss'], color=colors[0])
        ax1.tick_params(axis="y", labelcolor=colors[0])
    else:
        ax1.set_ylabel("Loss/Entropy scale", color=colors[0])
        ax1.plot(df['iteration'], df['loss'],
                 label="full loss", color=colors[0])
        ax1.plot(df['iteration'], df['hinge loss'],
                 label="hinge loss", color=colors[1])
        ax1.plot(df['iteration'], df['rl loss'],
                 label="rl loss", color=colors[2])
        ax1.plot(df['iteration'], df['entropy'],
                 label="entropy", color=colors[3])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax1.legend(handles, labels, loc='lower right')

    # all plots belonging to the second axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    # we already handled the x-label with ax1
    ax2.set_ylabel("Accuracy", color=color)

    ax2.plot(df['iteration'], df['accuracy'], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    # saving of figure
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='Path to the csv with metrics.')
    parser.add_argument('--rl', action='store_true',
                        help='Add when REINFORCE was used for training the sender')
    args = parser.parse_args()

    plot_data(args.path, args.rl)

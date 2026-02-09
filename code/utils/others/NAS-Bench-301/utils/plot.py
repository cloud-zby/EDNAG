import os
import numpy as np
import matplotlib.pyplot as plt


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_denoise(save_dir, avg_acc_trace, max_acc_trace, seed, dataset):
    create_dir(save_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    #
    ax1.plot(np.array(avg_acc_trace), color="blue", label="avg_acc")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.set_title("Average Accuracy Trace for Denoise")

    #
    ax2.plot(np.array(max_acc_trace), color="red", label="max_acc")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Max Accuracy Trace for Denoise")

    #
    plt.savefig(f"{save_dir}/{dataset}_seed_{seed}.png")
    plt.close()

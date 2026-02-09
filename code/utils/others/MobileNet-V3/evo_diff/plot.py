import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_denoise(
    save_dir,
    avg_acc_trace,
    max_acc_trace,
    valid_rate_trace,
    uniq_rate_trace,
    seed,
    dataset,
):

    create_dir(save_dir)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(32, 7))

    # 绘制平均适应度值变化图
    ax1.plot(np.array(avg_acc_trace), color="blue", label="avg_acc")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.set_title("Average Accuracy Trace for Denoise")

    # 绘制最大适应度值变化图
    ax2.plot(np.array(max_acc_trace), color="red", label="max_acc")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Max Accuracy Trace for Denoise")

    # 绘制有效率变化图
    ax3.plot(np.array(valid_rate_trace), color="orange", label="valid_rate")
    ax3.set_xlabel("Denoise Iteration")
    ax3.set_ylabel("Valid Rate")
    ax3.legend()
    ax3.set_title("Valid Rate Trace")

    # 绘制unique率变化图
    ax4.plot(np.array(uniq_rate_trace), color="green", label="uniq_rate")
    ax4.set_xlabel("Denoise Iteration")
    ax4.set_ylabel("Unique Rate")
    ax4.legend()
    ax4.set_title("Unique Rate Trace")

    # 保存图片
    plt.savefig(f"{save_dir}/{dataset}_seed_{seed}.png")
    plt.close()

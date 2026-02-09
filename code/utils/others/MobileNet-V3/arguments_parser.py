import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="EvoDiff in MobileNetV3")
    parser.add_argument(
        "--dataset",
        type=str,
        # default="cifar10",
        default="aircraft",
        help="Dataset in cifar10, cifar100, aircraft or pets",
    )
    parser.add_argument(
        "--experiment_num",
        type=int,
        default=10,
        help="Number of experiments to run",
    )
    parser.add_argument(
        "--num_step",
        type=int,
        default=100,
        help="Number of steps in the diffusion process",
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=100,
        help="Number of architectures in the population",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.8,
        help="Noise scale in the diffusion process",
    )
    parser.add_argument(
        "--mutate_rate",
        type=float,
        default=0.6,
        help="Mutation rate in the diffusion process",
    )
    parser.add_argument(
        "--diver_rate",
        type=float,
        default=0.2,
        help="Diversity rate in the diffusion process",
    )
    parser.add_argument(
        "--elite_rate",
        type=float,
        default=0.1,
        help="Elite rate in the diffusion process",
    )
    parser.add_argument(
        "--lower_params_rate",
        type=float,
        default=0.3,
        help="Rate of architectures with lower parameters",
    )
    parser.add_argument(
        "--mutate_distri_index",
        type=float,
        default=4,
        help="Mutation distribution index in the diffusion process",
    )
    parser.add_argument(
        "--plot_results",
        type=bool,
        default=True,
        help="Whether to plot the results",
    )
    parser.add_argument(
        "--max_iter_time",
        type=int,
        default=300,
        help="Maximum iteration time (seconds) for each experiment",
    )
    parser.add_argument(
        "--random_or_reproduce",
        type=str,
        default="reproduce",
        choices=["random", "reproduce"],
        help="Choose 'random' for random search or 'reproduce' to reproduce the results",
    )
    parser.add_argument(
        "--init_valid_rate",
        type=float,
        default=0.5,
        help="Initial valid rate in the population",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Top k architectures to select for training and evaluation",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="naive",
        choices=["mixed_prec", "naive", "accum_grad"],
        help="Training mode: 'mixed_prec' for mixed precision, 'naive' for standard training, 'accum_grad' for gradient accumulation",
    )
    # # Not used arguments for training
    # parser.add_argument(
    #     "--training_learning_rate",
    #     type=float,
    #     default=0.01,
    #     help="Learning rate for updating model parameters",
    # )
    # parser.add_argument(
    #     "--training_momentum",
    #     type=float,
    #     default=0.9,
    #     help="Momentum factor for accelerating gradient descent",
    # )
    # parser.add_argument(
    #     "--training_weight_decay",
    #     type=float,
    #     default=4e-5,
    #     help="Weight decay for regularization to prevent overfitting",
    # )
    # parser.add_argument(
    #     "--training_epochs",
    #     type=int,
    #     default=20,
    #     help="Number of epochs to train the model",
    # )
    # parser.add_argument(
    #     "--training_grad_clip",
    #     type=float,
    #     default=5.0,
    #     help="Maximum gradient norm for gradient clipping",
    # )
    # parser.add_argument(
    #     "--training_cutout_length",
    #     type=int,
    #     default=16,
    #     help="Length of cutout holes for data augmentation",
    # )
    # parser.add_argument(
    #     "--training_drop",
    #     type=float,
    #     default=0.2,
    #     help="Dropout rate for regularization",
    # )
    # parser.add_argument(
    #     "--training_drop_path",
    #     type=float,
    #     default=0.2,
    #     help="Drop path rate for regularization",
    # )
    # parser.add_argument(
    #     "--training_img_size",
    #     type=int,
    #     default=224,
    #     help="Input image size for the model",
    # )
    # parser.add_argument(
    #     "--training_batch_size",
    #     type=int,
    #     default=96,
    #     help="Number of samples per batch",
    # )
    return parser.parse_args()

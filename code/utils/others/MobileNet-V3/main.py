from experiment import random_exp, reproduce_exp
from arguments_parser import parse_arguments
from evo_diff.seed_config import EXP_SEED


def main():
    args = parse_arguments()
    if args.random_or_reproduce not in ["random", "reproduce"]:
        raise ValueError('random_or_reproduce must be either "random" or "reproduce"')
    if args.dataset not in ["cifar10", "cifar100", "aircraft", "pets"]:
        raise ValueError(
            'dataset must be either "cifar10", "cifar100", "aircraft", or "pets"'
        )
    if args.random_or_reproduce == "random":
        random_exp(
            experiment_num=args.experiment_num,
            dataset=args.dataset,
            num_step=args.num_step,
            population_num=args.population_num,
            top_k=args.top_k,
            noise_scale=args.noise_scale,
            mutate_rate=args.mutate_rate,
            diver_rate=args.diver_rate,
            elite_rate=args.elite_rate,
            lower_params_rate=args.lower_params_rate,
            mutate_distri_index=args.mutate_distri_index,
            plot_results=args.plot_results,
            max_iter_time=args.max_iter_time,
            init_valid_rate=args.init_valid_rate,
            train_mode=args.train_mode,
        )
    else:
        reproduce_exp(
            seed_list=EXP_SEED[args.dataset],
            dataset=args.dataset,
            num_step=args.num_step,
            population_num=args.population_num,
            top_k=args.top_k,
            noise_scale=args.noise_scale,
            mutate_rate=args.mutate_rate,
            diver_rate=args.diver_rate,
            elite_rate=args.elite_rate,
            lower_params_rate=args.lower_params_rate,
            mutate_distri_index=args.mutate_distri_index,
            plot_results=args.plot_results,
            max_iter_time=args.max_iter_time,
            init_valid_rate=args.init_valid_rate,
            train_mode=args.train_mode,
        )


if __name__ == "__main__":
    main()

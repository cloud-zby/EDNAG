import warnings
import argparse
warnings.filterwarnings('ignore')
from experiments import exp_with_fixed_seed_in_nb201, exp_with_rand_seed_in_nb201, exp_with_fixed_seed_in_meta_predictor, exp_with_rand_seed_in_meta_predictor

def main(args):
    dataset_name = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'imagenet': 'ImageNet16-120', 'aircraft': 'aircraft', 'pets': 'pets'}
    assert args.dataset.lower() in list(dataset_name.keys()), f'ERROR: invalid dataset {args.dataset}'
    assert args.exp_type.lower() in ['reproduce', 'random'], f'ERROR: invalid exp_type {args.exp_type}'

    if args.exp_type.lower() == 'reproduce':
        if args.dataset.lower() in ['cifar10', 'cifar100', 'imagenet']:
            exp_with_fixed_seed_in_nb201(dataset=dataset_name[args.dataset.lower()])
        else:
            exp_with_fixed_seed_in_meta_predictor(dataset=dataset_name[args.dataset.lower()])
    else:
        if args.dataset.lower() in ['cifar10', 'cifar100', 'imagenet']:
            exp_with_rand_seed_in_nb201(dataset=dataset_name[args.dataset.lower()])
        else:
            exp_with_rand_seed_in_meta_predictor(dataset=dataset_name[args.dataset.lower()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvoDiff-NAS')
    parser.add_argument('--exp_type', type=str, default='reproduce', help='reproduce, random')
    parser.add_argument('--dataset', type=str, help='cifar10, cifar100, imagenet, aircraft, pets')
    args = parser.parse_args()
    main(args)
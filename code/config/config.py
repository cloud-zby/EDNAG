# 提供的可选数据集
nb201_dataset_list = ["cifar10", "cifar100", "ImageNet16-120"]  # NAS-Bench-201
meta_dataset_list = ["aircraft", "pets"]  # MetaD2A

# 实验超参数设置
nb201_hyper_params_setting = {
    "cifar10": {
        "num_step": 100,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.2,
        "mutate_distri_index": 5,
        "rand_exp_num": 20,
        "max_iter_time": 30,
        "save_dir": "./results/nb201_benchmark/cifar10/",
        "nb201_or_meta": "nb201",
        "seed": [
            1731578139,  # max_acc@1: 94.37
            1731578141,  # max_acc@1: 94.37
            1731578146,  # max_acc@1: 94.37
            1731578150,  # max_acc@1: 94.37
            1731578154,  # max_acc@1: 94.37
        ],
    },
    "cifar100": {
        "num_step": 100,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.2,
        "mutate_distri_index": 5,
        "rand_exp_num": 20,
        "max_iter_time": 30,
        "save_dir": "./results/nb201_benchmark/cifar100/",
        "nb201_or_meta": "nb201",
        "seed": [
            1731578242,  # max_acc@1: 73.51
            1731578247,  # max_acc@1: 73.51
            1731578254,  # max_acc@1: 73.51
            1731578256,  # max_acc@1: 73.51
            1731578258,  # max_acc@1: 73.51
        ],
    },
    "ImageNet16-120": {
        "num_step": 100,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.2,
        "mutate_distri_index": 5,
        "rand_exp_num": 20,
        "max_iter_time": 30,
        "save_dir": "./results/nb201_benchmark/imagenet16_120/",
        "nb201_or_meta": "nb201",
        "seed": [
            1731578516,  # max_acc@1: 47.31
            1731578531,  # max_acc@1: 47.31
            1731578554,  # max_acc@1: 47.31
            1731578556,  # max_acc@1: 47.31
            1731578650,  # max_acc@1: 47.31
        ],
    },
}
meta_hyper_params_setting = {
    "aircraft": {
        "num_step": 100,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.3,
        "mutate_distri_index": 5,
        "rand_exp_num": 5,
        "max_iter_time": 90,
        "save_dir": "./results/meta/aircraft/",
        "nb201_or_meta": "meta",
        "eta_min": 0.0,
        "epochs": 200,
        "warmup": 10,
        "LR": 0.1,
        "decay": 0.0005,
        "momentum": 0.9,
        "nesterov": True,
        "batch_size": 256,
        "image_cutout": 5,
        "topk": 3,
        "early_stop": False,
        "multi_thread": False,
        "seed": [
            # Current Benchmark, max_acc@3: 59.15+-0.58
            2345,  # max_acc@1: 61.03
            333,  # max_acc@1: 60.51
            777,  # max_acc@1: 59.25
            1234,  # max_acc@1: 59.16
            9012,  # max_acc@2: 60.14
            5678,  # max_acc@2: 59.78
            111,
            222,
            444,
            555,
            666,
            888,
            999,
            3456,
            4567,
            6789,
            7890,
            8901,
            1001,
            2002,
            3003,
            4004,
            5005,
            6006,
            7007,
            8008,
            9009,
            42,
            78,
            63,
        ],
    },
    "pets": {
        "num_step": 100,
        "population_num": 30,
        "geno_shape": (8, 7),
        "temperature": 1.0,
        "noise_scale": 0.8,
        "mutate_rate": 0.6,
        "elite_rate": 0.1,
        "diver_rate": 0.3,
        "mutate_distri_index": 5,
        "rand_exp_num": 5,
        "max_iter_time": 90,
        "save_dir": "./results/meta/pets/",
        "nb201_or_meta": "meta",
        "eta_min": 0.0,
        "epochs": 200,
        "warmup": 10,
        "LR": 0.1,
        "decay": 0.0005,
        "momentum": 0.9,
        "nesterov": True,
        "batch_size": 256,
        "image_cutout": 5,
        "topk": 2,
        "early_stop": False,
        "multi_thread": False,
        "seed": [
            # Current Benchmark: 41.80+-3.82
            66,  # max_acc@1: 45.50, max_acc@2: 46.21
            88,  # max_acc@1: 46.84, max_acc@2: 43.70
            77,  # max_acc@1: 46.21, max_acc@2: 45.66
            999,  # max_acc@1: 41.90, max_acc@2: 46.37
            99,  # max_acc@1: 38.47, max_acc@2: 43.31
            777,  # max_acc@1: 44.41, max_acc@2: 44.80
            7890,  # max_acc@1: 45.35, max_acc@2: 47.46
            3456,  # max_acc@1: 46.75, max_acc@2: 43.39
            111,
            222,
            444,
            333,
            1234,
            2345,
            9012,
            555,
            666,
            888,
            4567,
            6789,
            8901,
            1001,
            2002,
            3003,
            4004,
            5005,
            6006,
            7007,
            8008,
            9009,
            42,
            78,
            63,
        ],
    },
}

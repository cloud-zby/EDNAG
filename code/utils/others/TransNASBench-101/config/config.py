# 实验超参数设置
hyper_params_setting = {
    "macro": {
        "class_scene": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/macro/class_scene",
            "seed": [
                0,  # 56.91
                2,  # 56.92
                3,  # 56.95
                5,  # 57.48
                6,  # 56.91
                15,  # 57.48
                16,  # 56.91
                17,  # 57.48
                40,  # 56.95
                888,  # 57.09
            ],
        },
        "class_object": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/macro/class_object",
            "seed": [
                0,  # 47.42
                1,  # 47.96
                3,  # 47.96
                5,  # 47.96
                7,  # 47.96
                9,  # 47.35
                11,  # 47.96
                12,  # 47.42
                17,  # 47.96
                18,  # 47.96
            ],
        },
        "room_layout": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/macro/room_layout",
            "seed": [
                0,  # 55.68
                4,  # 56.72
                11,  # 57.21
                14,  # 55.68
                30,  # 55.68
                50,  # 57.61
                80,  # 55.68
                22,  # 55.68
                55,  # 56.93
                66,  # 55.68
            ],
        },
        "jigsaw": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/macro/jigsaw",
            "seed": [
                0,  # 97.02
                1,  # 97.02
                2,  # 97.02
                3,  # 97.02
                4,  # 97.02
                5,  # 97.02
                6,  # 96.94
                7,  # 97.02
                8,  # 97.02
                13,  # 96.94
            ],
        },
        "segmentsemantic": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/macro/segmentsemantic",
            "seed": [
                1,  # 29.17
                2,  # 29.54
                6,  # 29.66
                7,  # 29.28
                8,  # 29.33
                12,  # 29.40
                13,  # 29.66
                17,  # 29.66
                18,  # 29.41
                50,  # 29.54
            ],
        },
        "normal": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/macro/normal",
            "seed": [
                0,  # 64.35
                1,  # 64.35
                2,  # 64.35
                3,  # 62.65
                10,  # 64.35
                12,  # 64.35
                14,  # 61.99
                16,  # 64.35
                30,  # 64.35
                50,  # 62.65
            ],
        },
        "autoencoder": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 5),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/macro/autoencoder",
            "seed": [
                0,  # 76.88
                1,  # 76.88
                2,  # 76.88
                3,  # 74.91
                5,  # 74.76
                6,  # 76.88
                7,  # 76.88
                8,  # 73.99
                9,  # 76.88
                10,  # 76.88
            ],
        },
    },
    "micro": {
        "class_scene": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/micro/class_scene",
            "seed": [
                0,  # 54.86
                1,  # 54.94
                2,  # 54.94
                3,  # 54.87
                4,  # 54.93
                6,  # 54.79
                7,  # 54.86
                11,  # 54.94
                12,  # 54.94
                17,  # 54.94
            ],
        },
        "class_object": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/micro/class_object",
            "seed": [
                1,  # 46.06
                3,  # 46.32
                4,  # 46.23
                5,  # 46.32
                10,  # 46.32
                11,  # 46.32
                13,  # 46.32
                14,  # 46.23
                20,  # 46.32
                30,  # 46.32
            ],
        },
        "room_layout": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/micro/room_layout",
            "seed": [
                2,  # 59.31
                4,  # 58.71
                5,  # 59.31
                6,  # 58.71
                10,  # 58.71
                14,  # 58.71
                15,  # 58.71
                17,  # 59.94
                19,  # 58.71
                30,  # 58.71
            ],
        },
        "jigsaw": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/micro/jigsaw",
            "seed": [
                0,  # 95.37
                1,  # 94.91
                2,  # 94.99
                3,  # 95.00
                5,  # 94.87
                6,  # 95.37
                9,  # 95.37
                10,  # 95.37
                11,  # 95.37
                12,  # 95.37
            ],
        },
        "segmentsemantic": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/micro/segmentsemantic",
            "seed": [
                0,  # 26.10
                1,  # 25.95
                3,  # 26.27
                4,  # 26.27
                5,  # 26.27
                8,  # 25.91
                9,  # 26.27
                10,  # 26.27
                14,  # 26.27
                18,  # 25.80
            ],
        },
        "normal": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/micro/normal",
            "seed": [
                0,  # 58.89
                4,  # 59.62
                8,  # 58.23
                9,  # 57.84
                10,  # 58.89
                11,  # 59.62
                12,  # 59.62
                18,  # 58.73
                20,  # 59.62
                50,  # 59.62
            ],
        },
        "autoencoder": {
            "num_step": 100,
            "population_num": 30,
            "geno_shape": (6, 4),
            "temperature": 1.0,
            "noise_scale": 0.8,
            "mutate_rate": 0.6,
            "elite_rate": 0.1,
            "diver_rate": 0.2,
            "mutate_distri_index": 5,
            "rand_exp_num": 20,
            "max_iter_time": 30,
            "save_dir": "./results/micro/autoencoder",
            "seed": [
                0,  # 57.19
                1,  # 57.53
                3,  # 57.72
                6,  # 57.53
                7,  # 57.72
                9,  # 57.26
                12,  # 57.72
                13,  # 57.72
                15,  # 57.72
                16,  # 57.72
            ],
        },
    },
}

# EDNAG
This repository contains the code for the paper "Evolution Meets Diffusion: Efficient Neural Architecture Generation" (EDNAG: **E**volutionary **D**iffusion-based **N**eural **A**rchitecture **G**eneration).         

## Overview
![overview](method_overview.png)

Neural Architecture Search (NAS) has gained widespread attention for its transformative potential in deep learning model design. However, the vast and complex search space of NAS leads to significant computational and time costs. Neural Architecture Generation (NAG) addresses this by reframing NAS as a generation problem, enabling the precise generation of optimal architectures for specific tasks. Despite its promise, mainstream methods like diffusion models face limitations in global search capabilities and are still hindered by high computational and time demands. When migrating to new search spaces, current methods require retraining, which incurs substantial computational overhead and makes it difficult to quickly adapt to unseen search spaces. To overcome these challenges, we propose Evolutionary Diffusion-based Neural Architecture Generation (EDNAG), a novel approach that achieves efficient and training-free architecture generation. EDNAG leverages evolutionary algorithms to simulate the denoising process in diffusion models, using fitness to guide the transition from random Gaussian distributions to optimal architecture distributions. This approach combines the strengths of evolutionary strategies and diffusion models, enabling rapid and effective architecture generation. Extensive experiments demonstrate that EDNAG achieves state-of-the-art (SOTA) performance in architecture optimization, with an improvement in accuracy of up to 10.45%. Furthermore, it eliminates the need for time-consuming training and boosts inference speed by an average of 50×, showcasing its exceptional efficiency and effectiveness.

![Denoise](Denoise.png)

- You can find the paper [here](https://arxiv.org/abs/2504.17827).

## Code Snippets
Using the code for NAS-Bench-201 search space as an example:
- `main.py`: The entry point of the program, which parses arguments and calls `experiments.py`.
- `experiments.py`: The pipeline for experiments, setting random seeds and providing the workflow for `generation`-`get_topk_archs`-`evaluate_archs`.
- `evo_diff.py`: Neural architecture generation, implementing the pipeline of the Fitness-guided Denoising (FD) Strategy.
- `ddim.py` and `predictor.py`: DDIM denoising process in the FD strategy.
- `corrector.py`: Selection-based optimization strategy in EDNAG.
- `mapping.py`: Definition and implementation of Fitness-to-Probability Mapping functions.
- `analyse.py`: Conversion between two neural architecture representations, matrix and string.
- `meta_fitness.py`: Fitness evaluation by a dataset-aware meta neural predictor.
- `coreset.py`: Core dataset image selection for the neural predictor.
- `nb201_fitness.py`: Get architecture accuracy by querying the NAS-Bench-201 benchmark table.
- `eval_arch.py`: Evaluate architectures by training and testing.
- `config.py`: Configurations for experiment parameters and random seeds.
  
#### Detailed Explaination
Step-by-step denoising iterations in the code.
```python
scheduler = DDIMSchedulerCosine(num_step=100)
for t, alpha in scheduler:
    fitness = two_peak_density(x, std=0.25)
    print(f"fitness {fitness.mean().item()}")
    # apply the power mapping function
    generator = BayesianGenerator(x, mapping_fn(fitness), alpha)
    x = generator(noise=0.1)
    trace.append(x)
```

Get the predicted $\hat{x_0}$ using the estimator function.
```python
def generate(self, noise=1.0, return_x0=False):
    x0_est = self.estimator(self.x)
    x_next = ddim_step(self.x, x0_est, (self.alpha, self.alpha_past), noise=noise)
    if return_x0:
        return x_next, x0_est
    else:
        return x_next
```

Perform iterative denoising using the formula $x_{t-1} = \sqrt{\alpha_{t-1}} \cdot \hat{x_0} + \sqrt{1 - \alpha_{t-1} - \sigma^2_t} \cdot \hat{\epsilon} + \sigma_t w$.
```python
def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    alphat, alphatp = alphas
    sigma = ddpm_sigma(alphat, alphatp) * noise
    eps = (xt - (alphat ** 0.5) * x0) / (1.0 - alphat) ** 0.5
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (alphatp ** 0.5) * x0 + ((1 - alphatp - sigma ** 2) ** 0.5) * eps + sigma * torch.randn_like(x0)
    return x_next
```

$\sigma$ is calculated by the variance of $x_t$, $\sigma = \sqrt{\frac{(1 - \alpha_{t'})}{(1 - \alpha_t)} \cdot \left(1 - \frac{\alpha_t}{\alpha_t}\right)}$, which the author notes is the default formula used in DDPM.
```python
def ddpm_sigma(alphat, alphatp):
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5
```

Balance of Convergence and Diversity：

`elite_rate` ensures the best samples are retained during iterations, ensuring convergence. `diver_rate` ensures diverse samples are retained during iterations, ensuring diversity.
`mutate_rate` controls the number of samples that undergo mutation, and `mutate_distri_index` controls the scale of mutation to ensure exploration of the solution space.

## Experiments

Here we give an example of how to run experiments on NAS-Bench-201 search space. For other search spaces, please refer to the code and modify the dataset and fitness evaluation accordingly.

Before experiments, download following 11 dataset files.  

<h4 style="color:gray">1. AIRCRAFT</h4>

#### Save Path
./meta_acc_predictor/data/fgvc-aircraft-2013b
#### Download URL
- https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz

<h4 style="color:gray">2. PETS</h4>  

#### Save Path
./meta_acc_predictor/data/pets/
#### Download URL
- https://www.dropbox.com/s/kzmrwyyk5iaugv0/test15.pth?dl=1
- https://www.dropbox.com/s/w7mikpztkamnw9s/train85.pth?dl=1

<h4 style="color:gray">3. NASBENCH201</h4>

#### Save Path
./meta_acc_predictor/data/nasbench201.pt
#### Download URL
- https://github.com/CownowAn/DiffusionNAG/blob/main/NAS-Bench-201/data/transfer_nag/nasbench201.pt

<h4 style="color:gray">4. META_PREDICTOR</h4>  

#### Save Path
./meta_acc_predictor/unnoised_checkpoint.pth.tar
#### Download URL
- https://drive.google.com/file/d/1S2IV6L9t6Hlhh6vGsQkyqMJGt5NnJ8pj/view?usp=sharing

<h4 style="color:gray">5. CIFAR10_BY_LABEL</h4>  

#### Save Path
./meta_acc_predictor/data/meta_predictor_dataset/cifar10bylabel.pt
#### Download URL
- https://www.dropbox.com/s/wt1pcwi991xyhwr/cifar10bylabel.pt?dl=1

<h4 style="color:gray">6. CIFAR100_BY_LABEL</h4>  

#### Save Path
./meta_acc_predictor/data/meta_predictor_dataset/cifar100bylabel.pt
#### Download URL
- https://www.dropbox.com/s/nn6mlrk1jijg108/aircraft100bylabel.pt?dl=1

<h4 style="color:gray">7. AIRCRAFT_BY_LABEL</h4>  

#### Save Path
./meta_acc_predictor/data/meta_predictor_dataset/aircraftbylabel.pt
#### Download URL
- https://www.dropbox.com/s/nn6mlrk1jijg108/aircraft100bylabel.pt?dl=1

<h4 style="color:gray">8. PETS_BY_LABEL</h4>  

#### Save Path
./meta_acc_predictor/data/meta_predictor_dataset/petsbylabel.pt
#### Download URL
- https://www.dropbox.com/s/mxh6qz3grhy7wcn/petsbylabel.pt?dl=1

<h4 style="color:gray">9. IMAGENET32_BY_LABEL</h4>  

#### Save Path
./meta_acc_predictor/data/meta_predictor_dataset/imgnet32bylabel.pt
#### Download URL
- https://www.dropbox.com/s/7r3hpugql8qgi9d/imgnet32bylabel.pt?dl=1

<h4 style="color:gray">10. NAS_BENCH_201_API_V1.0</h4>  

#### Save Path
./nas_201_api/NAS-Bench-201-v1_0-e61699.pth
#### Download URL
- https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs

<h4 style="color:gray">11. NAS_BENCH_201_API_V1.3</h4>  

#### Save Path
./nas_201_api/NAS-Bench-201-v1_1-096897.pth
#### Download URL
- https://drive.google.com/open?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_

<h4 style="color:gray">Install</h4>

```
cd NAS-Bench-201
conda create -n evo_diff python=3.12.2
pip install -r requirements.txt
```

Note: If you use PyTorch >= 2.6, you should add `weights_only=False` for every `torch.load()`, because the default value of the `weights_only` argument in `torch.load` is changed from `False` to `True` in PyTorch 2.6. 

<h4 style="color:gray">Reproduce the results</h4>

```
python ./main.py --dataset cifar10
python ./main.py --dataset cifar100
python ./main.py --dataset imagenet
python ./main.py --dataset aircraft
python ./main.py --dataset pets
```

<h4 style="color:gray">Run random experiments</h4>

```
python ./main.py --exp_type random --dataset cifar10
python ./main.py --exp_type random --dataset cifar100
python ./main.py --exp_type random --dataset imagenet
python ./main.py --exp_type random --dataset aircraft
python ./main.py --exp_type random --dataset pets
```



<h4 style="color:gray">Architecture Examples</h4>

Nodes represent operations, while adjacency matrices are fixed. Therefore, it is only necessary to generate a matrix of node operations. The size of the operation matrix is (number of node layers, number of candidate operands). 
Since `ops = ['input', 'output', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']`, there are 8 layers of nodes and 7 layers of operation. 

<h4 style="color:gray">Experiments Results</h4>

![nb201](NAS-Bench-201/exp_result.png)


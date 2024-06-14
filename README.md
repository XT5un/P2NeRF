# [CVPR2024] Global and Hierarchical Geometry Consistency Priors for Few-shot NeRFs in Indoor Scenes

## Installation

We use cuda-11.1 in our experiments.

``` sh
conda create -n p2nerf python=3.6.15
conda activate p2nerf
pip install --upgrade pip
pip install --upgrade jaxlib==0.1.68+cuda111 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

## Data

You can download the data we processed from [here]([http://](https://drive.google.com/file/d/1GzWVmp1PLUL4XCqKCFB0-WjUqzoV1xyp/view?usp=drive_link)) and unzip the downloaded zip into the `data/` folder.

## Running

### Scannet dataset

``` sh
chmod +x ./scripts/ddp.sh
./scripts/ddp.sh
```

### Replica dataset

``` sh
chmod +x ./scripts/replica.sh
./scripts/replica.sh
```

## Citation

``` text
@InProceedings{Sun2024P2NeRF,
    author = {Xiaotian Sun and Qingshan Xu and Xinjie Yang and Yu Zang and Cheng Wang}, 
    title = {Global and Hierarchical Geometry Consistency Priors for Few-shot NeRFs in Indoor Scenes},
    booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024},
}
```

## Acknowledgments

This code heavily references the `FreeNeRF` and `RegNeRF` codebases, and the authors are thanked for their open source behaviour.

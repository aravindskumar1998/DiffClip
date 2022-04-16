# DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation (CVPR 2022) 

This is a modification to the original PyTorch implementation of DiffusionCLIP to enable easier dataloading.

## Getting Started

### Installation
We recommend running our code using:

- NVIDIA GPU + CUDA, CuDNN
- Python 3, Anaconda

To install our implementation, clone our repository and run following commands to install necessary packages:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
### Resources
- For the original fine-tuning, VRAM of 24 GB+ for 256x256 images are required.  
- For the GPU-efficient fine-tuning, VRAM of 12 GB+ for 256x256 images and 24 GB+ for 512x512 images are required.   
- For the inference, VRAM of 6 GB+ for 256x256 images and 9 GB+ for 512x512 images are required.  

### Pretrained Models for DiffusionCLIP Fine-tuning

To manipulate soure images into images in CLIP-guided domain, the **pretrained Diffuson models** are required.

| Image Type to Edit |Size| Pretrained Model | Dataset | Reference Repo. 
|---|---|---|---|---
| Human face |256×256| Diffusion (Auto), [IR-SE50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view) | [CelebA-HQ](https://arxiv.org/abs/1710.10196) | [SDEdit](https://github.com/ermongroup/SDEdit), [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 
| Church |256×256| Diffusion (Auto) | [LSUN-Bedroom](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Bedroom |256×256| Diffusion (Auto) | [LSUN-Church](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Dog face |256×256| [Diffusion](https://drive.google.com/file/d/14OG_o3aa8Hxmfu36IIRyOgRwEP6ngLdo/view) | [AFHQ-Dog](https://arxiv.org/abs/1912.01865) | [ILVR](https://github.com/jychoi118/ilvr_adm)
| ImageNet |512×512| [Diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt) | [ImageNet](https://image-net.org/index.php) | [Guided Diffusion](https://github.com/openai/guided-diffusion)


### Datasets 
To precompute latents and fine-tune the Diffusion models, you need about 30+ images in the source domain. Download them and fill their paths in `./configs/paths_config.py`. 

## DiffusionCLIP Fine-tuning 


To fine-tune the pretrained Diffusion model guided by CLIP, run the following commands:

```
python main.py --clip_finetune          \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr neanderthal  \
               --do_train 1             \
               --do_test 1              \
               --n_train_img 50         \
               --n_test_img 10          \
               --n_iter 5               \
               --t_0 500                \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 40         \
               --lr_clip_finetune 8e-6  \
               --id_loss_w 0            \
               --l1_loss_w 1            
```
- You can use `--clip_finetune_eff` instead of `--clip_finetune` to save GPU memory.
- `config`: `celeba.yml` for human face, `bedroom.yml` for bedroom, `church.yml` for church, `afhq.yml` for dog face and , `imagenet.yml` for images from ImageNet.
- `exp`: Experiment name.
- `edit_attr`: Attribute to edit, you can use `./utils/text_dic.py` to predefined source-target text pairs or define new pair. 
  - Instead, you can use `--src_txts` and `--trg_txts`. 
- `do_train`, `do_test`: If you finish training quickly withouth checking the outputs in the middle of training, you can set `do_test` as 1.
- `n_train_img`, `n_test_img`: # of images in the trained domain for training and test.        
- `n_iter`: # of iterations of a generative process with `n_train_img` images.
- `t_0`: Return step in [0, 1000), high `t_0` enable severe change but may lose more identity or semantics in the original image.  
- `n_inv_step`, `n_train_step`, `n_test_step`: # of steps during the generative pross for the inversion, training and test respectively. They are in `[0, t_0]`. We usually use 40, 6 and 40  for `n_inv_step`, `n_train_step` and `n_test_step` respectively. 
   - We found that the manipulation quality is better when `n_***_step` does not divide `t_0`. So we usally use 301, 401, 500 or 601 for `t_0`.
- `lr_clip_finetune`: Initial learning rate for CLIP-guided fine-tuning.
- `id_loss_w`, `l1_loss` : Weights of ID loss and L1 loss when CLIP loss weight is 3.



## Novel Applications

The fine-tuned models through DiffusionCLIP can be leveraged to perform the several novel applications. 

### Manipulation of Images in Trained Domain & to Unseen Domain
![](imgs/app_1_manipulation.png)

You can edit one image into the CLIP-guided domain by running the following command:
``` 
python main.py --edit_one_image            \
               --config celeba.yml         \
               --exp ./runs/test           \
               --t_0 500                   \
               --n_inv_step 40             \
               --n_test_step 40            \
               --n_iter 1                  \
               --img_path imgs/celeb1.png  \
               --model_path  checkpoint/neanderthal.pth
```
- `img_path`: Path of an image to edit
- `model_path`: Finetuned model path to use

You can edit multiple images from the dataset into the CLIP-guided domain by running the following command:
```
python main.py --edit_images_from_dataset  \
               --config celeba.yml         \
               --exp ./runs/test           \
               --n_test_img 50             \
               --t_0 500                   \
               --n_inv_step 40             \
               --n_test_step 40            \
               --model_path checkpoint/neanderthal.pth
```
 

### Image Translation from Unseen Domain into Another Unseen Domain
![](imgs/app_2_unseen2unseen.png)

## Related Works

Usage of guidance by [CLIP](https://arxiv.org/abs/2103.00020) to manipulate images is motivated by [StyleCLIP](https://arxiv.org/abs/2103.17249) and [StyleGAN-NADA](https://arxiv.org/abs/2108.00946).
Image translation from an unseen domain to the trained domain using diffusion models is introduced in [SDEdit](https://arxiv.org/abs/2108.01073), [ILVR](https://arxiv.org/abs/2108.02938).
DDIM sampling and its reveral for generation and inversion of images are introduced by in [DDIM](https://arxiv.org/abs/2010.02502), [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233).

Our code strcuture is based on the official codes of [SDEdit](https://github.com/ermongroup/SDEdit) and [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada). We used pretrained models from [SDEdit](https://github.com/ermongroup/SDEdit) and [ILVR](https://github.com/jychoi118/ilvr_adm).


## Citation
If you find DiffusionCLIP useful in your research, please consider citing:

    @article{kim2021diffusionclip,
      title={Diffusionclip: Text-guided image manipulation using diffusion models},
      author={Kim, Gwanghyun and Ye, Jong Chul},
      journal={arXiv preprint arXiv:2110.02711},
      year={2021}
    }

# TransUNet

This is an implementation of the paper [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306) with slightly modifications. For the official implementation, check out this [repo](https://github.com/Beckschen/TransUNet).

## Modifications

1. **Upsampling** \
   In the official implementation, authors use `nn.UpsamplingBilinear2d(scale_factor=2)`, but it's replaced with `nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)`.
2. **Number of heads in ViT** \
   Change from 12 to 16.
3. **Patch size** \
   According to the paper, the width and height at the bottom layer should be 1/8 and 1/16 of the image size at the input and output of ViT, which is 224/8 = 28 and 224/16 = 14. Therefore, we use 2 pixels as the patch size.
4. **Output size** \
   Use `torch.nn.functional.interpolate` to enlarge the output size to the input one.

## Objects

1. `TransUNetEncoderConvBlock`
2. `TransUNetViT`
3. `TransUNetEncoder`
4. `TransUNet`

## Usage

```python
from comvex.transunet import TransUNet

transUnet = TransUNet(
    input_channel=3,                      # Image channel
    middle_channel=512,                   # Latent channel at the bottom of the TransUNet, same as ViT's token dimension
    output_channel=2,                     # Number of classes
    channel_in_between=[64, 128, 256],    # Channel in the encoder and decoder, the length of `channel_in_between` means the number of layers.
    num_res_blocks_in_between=[3, 4, 9],  # Number of ResNet blocks in each layer of the encoder
    image_size=224,
    patch_size=2,                         # ViT's patch size
    dim=512,                              # ViT's token dimension
    num_heads=16,
    num_layers=12,                        # ViT's layers
    token_dropout=0,                      # ViT's token dropout
    ff_dropout=0,
    to_remain_size=True                   # True to interpolate the output size as the input
)
```

## Demo

```bash
python examples/TransUNet/demo.py
```

## Citation

```bibtex
@misc{chen2021transunet,
      title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
      author={Jieneng Chen and Yongyi Lu and Qihang Yu and Xiangde Luo and Ehsan Adeli and Yan Wang and Le Lu and Alan L. Yuille and Yuyin Zhou},
      year={2021},
      eprint={2102.04306},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

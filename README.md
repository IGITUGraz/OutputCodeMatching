# Improving Robustness Against Stealthy Weight Bit-Flip Attacks by Output Code Matching

This is the code repository of the following [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ozdenizci_Improving_Robustness_Against_Stealthy_Weight_Bit-Flip_Attacks_by_Output_Code_CVPR_2022_paper.pdf) to train deep neural networks with output code matching (OCM) to improve robustness against stealthy adversarial weight bit-flip attacks.

"Improving Robustness Against Stealthy Weight Bit-Flip Attacks by Output Code Matching"\
<em>Ozan Özdenizci, Robert Legenstein</em>\
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Currently the repository supports training the networks presented in the paper, and evaluating these networks with state-of-the-art [Stealthy T-BFA](https://arxiv.org/pdf/2007.12336.pdf) and [Stealthy TA-LBF](https://arxiv.org/pdf/2102.10496.pdf) attacks.

## Setup

You will need [PyTorch](https://pytorch.org/get-started/) to run this code. You can simply start by executing:
```bash
pip install -r requirements.txt
```
to install all dependencies and use the repository.

## Usage

You can use `main.py` to train and perform benign evaluations of quantized networks. Description of important arguments:

- `--dataset`: "CIFAR10", "CIFAR100", "ImageNet"
- `--arch`: "resnet20_quan", "resnet50_quan", "wrn28_4_quan", "wrn28_8_quan"
- `--bits`: quantization bits which is set to 8 or 4 in the paper
- `--ocm`: enable output code matching in the final layer of the model
- `--code_length`: length of the code bit strings for OCM (as a power of 2)
- `--output_act`: "linear", "tanh" (should be "tanh" for OCM models)

### End-to-end training with OCM & finetuning pre-trained vanilla models with OCM

- `ocm_train_cifar10.sh`: End-to-end training of ResNet-20 models on CIFAR-10 with OCM<sub>16</sub> and OCM<sub>64</sub>.
- `ocm_train_imagenet.sh` Training a vanilla ResNet-50 model on ImageNet and finetuning with OCM<sub>1024</sub>.

## Saved model weights

We share the OCM models trained on CIFAR-10 and ImageNet that are used for evaluations in the paper.
Different evaluations may naturally result in slight differences in the presented numbers.

* CIFAR-10 with ResNet-20 (8-bit): 
[OCM<sub>16</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinCVPR2022/resnet20_quan8_OCM16.zip) | 
[OCM<sub>64</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinCVPR2022/resnet20_quan8_OCM64.zip)
* ImageNet with ResNet-50 (8-bit):
[OCM<sub>1024</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinCVPR2022/resnet50_quan8_OCM1024.zip) | 
[OCM<sub>2048</sub>](https://igi-web.tugraz.at/download/OzdenizciLegensteinCVPR2022/resnet50_quan8_OCM2048.zip)

### An example on how to evaluate saved model weights

To evaluate the ResNet-50 models with OCM<sub>1024</sub> against stealthy T-BFA:
```bash
python attack_tbfa.py --data_dir "data/" --dataset "ImageNet" -c 1000 --arch "resnet50_quan" --bits 8 --ocm --code_length 1024 --output_act "tanh" --outdir "results/imagenet/resnet50_quan8_OCM1024/"
```

## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@inproceedings{ozdenizci2022cvpr,
  title={Improving robustness against stealthy weight bit-flip attacks by output code matching},
  author={Ozan \"{O}zdenizci and Robert Legenstein},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={13388-13397},
  year={2022}
}
```

## Acknowledgments

Authors of this work are affiliated with Graz University of Technology, Institute of Theoretical Computer Science, and Silicon Austria Labs, TU Graz - SAL Dependable Embedded Systems Lab, Graz, Austria. This work has been supported by the "University SAL Labs" initiative of Silicon Austria Labs (SAL) and its Austrian partner universities for applied fundamental research for electronic based systems.

Parts of this code repository is based on the following works:

* https://github.com/adnansirajrakin/T-BFA
* https://github.com/jiawangbai/TA-LBF
* https://github.com/elliothe/BFA

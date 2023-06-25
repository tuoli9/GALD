# GALD-Attack

This repository is the official repository for our paper "Improving Transferable Adversarial Attack for Vision Transformers via Global Attention and Local Drop".

## Datasets

Download the dataset from the following link and extract images to the path “./data/”:

https://drive.google.com/file/d/1VAELrOwK2uE7XBqrVpa8SZJmX2X52Abc/view?usp=sharing

## Experiments

You can run the following command to perform the GALD attack method, using vit_base_patch16_224 model as a surrogate model.

```python
python main.py --white vit --ra 0.7 --rd 0.1 
```

Please refer to 'main.sh` for more experiments.
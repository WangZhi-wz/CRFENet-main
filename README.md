# CRFENet:Coupled refinement and focus exploration network for medical image
segmentation
This repository contains the official Pytorch implementation of training & evaluation code for CRFENet.
[CRFENet:Coupled refinement and focus exploration network for medical image](111).
### Environment
- Install `CUDA 12.1` , `pytorch 1.7.1` and `python 3.8.10`
- Install other requirements: `pip install -r requirements.txt`

### Data
```
$ data
train
├── images
├── masks
valid
├── images
├── masks
test
├── images
├── masks
```

### 1. set up parameters

```bash
train.py
    parser.add_argument('--num_epochs', type=int, default=30, help='epoch number')
    parser.add_argument('--backbone', type=str, default='b3', help='backbone version')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--init_trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--data_path', type=str, default='CVC_ClinicDB811', help='path to dataset')
    ...
    ...
```

### 2. Training

```bash
train.py 
```

###  3. Testing

```bash
test.py


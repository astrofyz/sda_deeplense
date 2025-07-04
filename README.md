# Domain adaptation in application to gravitational lens finding

This repository contains the implementation of three domain adaptation methods for automated gravitational lens detection explored in https://arxiv.org/abs/2410.01203.

## Overview

The next decade is expected to see a tenfold increase in the number of strong gravitational lenses, driven by new wide-field imaging surveys. To discover these rare objects, efficient automated detection methods need to be developed. 
This project assesses the performance of three domain adaptation techniques ([ADDA](https://arxiv.org/abs/1702.05464), [WDGRL](https://arxiv.org/abs/1707.01217), and [SDA](https://arxiv.org/abs/1709.10190)) combined with ResNet-18 and ENN-based encoders in enhancing lens-finding algorithms trained on simulated data when applied to real observational data.

**Key Findings:**
- WDGRL combined with an ENN-based encoder provides the best performance in an unsupervised setting
- Supervised domain adaptation enhances the model's ability to distinguish between lenses and common similar-looking false positives, such as spiral galaxies

## Data

As a source dataset for domain adaptation we used simulations of strong gravitational lenses made with [lenstronomy](https://lenstronomy.readthedocs.io/en/latest/), as a target dataset we utilised strong lens candidates compiled from the literature, that were available in the Hyper Suprime-Cam Subaru Strategic Program (HSC-SSP) PDR2 Wide field. Datasets used in the paper are available from the corresponding author upon reasonable request.

### Dependencies
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib astropy
pip install e2cnn  # For equivariant neural networks
pip install tqdm
```

## Usage

### Basic Training Pipeline

For minimal training of the source model, performing domain adaptation with all three methods, and evaluating the model with the default parameters:

```
python main.py --encoder='ENN' --DA_methods="adda/sda/wdgrl" --out_dir="Output" --data_dir='../Data' --train_source --adapt --evaluate 
```

For the experiment including spiral contaminants add ```--include_spirals``` keyword

## Key Parameters

### Model Configuration
- `--encoder`: Choose encoder architecture (ResNet/ENN)
- `--DA_methods`: Domain adaptation methods to use (any of adda/wdgrl/sda or "all")

### Training Parameters
- `--n_epochs_enc`: Source model training epochs
- `--adda_epochs`: ADDA adaptation epochs
- `--wdgrl_epochs`: WDGRL adaptation epochs
- `--sda_epochs`: SDA adaptation epochs
- `--batch_size`: Training batch size
- `--lr_source`: Learning rate for source training
- `--lr_adda_critic/enc`: Learning rates for ADDA
- `--lr_wdgrl_critic/clf/encoder`: Learning rates for WDGRL
- `--lr_sda`: Learning rate for SDA

### Dataset Configuration
- `--src_size`: Number of positive examples in source dataset (simulated lenses)
- `--src_neg_size`: Number of negative examples in source dataset (defaults to src_size if not provided)
- `--tgt_size`: Number of positive examples in target dataset (real HSC-SSP lenses)
- `--tgt_neg_size`: Number of negative examples in target dataset (defaults to tgt_size if not provided)
- `--test_pos_size`: Number of positive examples in test dataset 
- `--test_neg_size`: Number of negative examples in test dataset 


### Evaluation
- `--desired_fpr`: Desired false positive rate for evaluation
- `--save_losses`: Save training losses for analysis

## Output Structure

The training process generates:
```
Output/
├── ENN/
│   ├── weights/
│   │   ├── source/
│   │   ├── adda/
│   │   ├── wdgrl/
│   │   └── sda/
│   ├── losses/
│   ├── evaluation_results/
│       ├── roc_curves.png
├── execution_log_YYYYMMDD_HHMMSS.txt
├── training_log_YYYYMMDD_HHMMSS.txt
```

## Contact

For questions and support, please contact [hparul@crimson.ua.edu]. 

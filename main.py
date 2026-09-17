import argparse
from util_module import *
from preprocess_module import *
from read_module import *
from train_module import *
from torchvision import transforms # type: ignore
import os
import logging
import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train on source')
    
    # Model architecture
    parser.add_argument('--encoder', type=str, default='ResNet', help='encoder model: ResNet/EffNet/ENN')
    parser.add_argument('--DA_methods', type=str, default="all", help="DA methods, e.g. sda/adda/wdgrl/all")

    # Directory and file paths
    parser.add_argument('--out_dir', type=str, default="", help="output directory for Weights and Results; error if doesn't exist")
    parser.add_argument('--data_dir', type=str, default='../Data', help="input directory with Data; error if doesn't exist")
    parser.add_argument('--savename', type=str, default="", help="suffix of Results savename (like _aug or _hsmokes etc)")
    
    # Dataset parameters
    parser.add_argument('--src_dataset', type=str, default='custom', help="simulations for source dataset") # this is not used anymore, but was required when we tested on Metcalf+2019 dataset
    parser.add_argument('--subsample', type=str, default='', help='subsample of lenses: S1/S2/S3')  # this is not used anymore and corresponds to the subsamples in the Metcalf+2019 dataset
    parser.add_argument('--simulated_test', action='store_true', help="Use simulated data for testing")  # this need to be added to read_module
    
    # Preprocessing
    parser.add_argument('--preprocess', type=str, default='none', help="preprocessing (none, sqrt, log)")

    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--min_delta', type=float, default=0.0001, help="Minimum delta for early stopping")
    parser.add_argument('--save_interval', type=int, default=10, help="Interval for saving models")
    
    # Training hyperparameters
    parser.add_argument('--adda_epochs', type=int, default=20, help="number of epochs for adda training")
    parser.add_argument('--wdgrl_epochs', type=int, default=20, help="number of epochs for wdgrl training")
    parser.add_argument('--sda_epochs', type=int, default=20, help="number of epochs for sda training")
    parser.add_argument('--n_epochs_enc', type=int, default=10, help="number of epochs during source training")
    parser.add_argument('--adda_n_encoder_steps', type=int, default=3, help="How many steps to update the encoder in ADDA")
    parser.add_argument('--wdgrl_n_critic_steps', type=int, default=5, help="How many steps to update the critic in WDGRL")
    parser.add_argument('--wdgrl_n_clf_steps', type=int, default=1, help="How many steps to update the classifier in WDGRL")
    
    # Learning rates
    parser.add_argument('--lr_source', type=float, default=1e-5, help="Learning rate for source training")
    parser.add_argument('--lr_source_scheduler', type=float, default=0.002, help="Learning rate in 1Cr scheduler for source training")
    parser.add_argument('--lr_adda_critic', type=float, default=1e-5, help="Learning rate for ADDA critic")
    parser.add_argument('--lr_adda_enc', type=float, default=1e-6, help="Learning rate for ADDA encoder")
    parser.add_argument('--lr_wdgrl_critic', type=float, default=1e-4, help="Learning rate for WDGRL critic")
    parser.add_argument('--lr_wdgrl_enc', type=float, default=1e-4, help="Learning rate for WDGRL encoder")
    parser.add_argument('--lr_wdgrl_clf', type=float, default=1e-4, help="Learning rate for WDGRL classifier")
    parser.add_argument('--lr_wdgrl_encoder', type=float, default=1e-4, help="Learning rate for WDGRL encoder")
    parser.add_argument('--lr_sda', type=float, default=1e-3, help="Learning rate for SDA")

    # Weight decay parameters
    parser.add_argument('--weight_decay_source', type=float, default=1e-6, help="Weight decay for source training")
    parser.add_argument('--weight_decay_adda_critic', type=float, default=2.5e-4, help="Weight decay for ADDA critic")
    parser.add_argument('--weight_decay_adda_enc', type=float, default=2.5e-5, help="Weight decay for ADDA encoder")
    parser.add_argument('--weight_decay_wdgrl_critic', type=float, default=1e-6, help="Weight decay for WDGRL critic")
    parser.add_argument('--weight_decay_wdgrl_clf', type=float, default=1e-6, help="Weight decay for WDGRL classifier")
    parser.add_argument('--weight_decay_wdgrl_encoder', type=float, default=1e-6, help="Weight decay for WDGRL encoder")
    parser.add_argument('--weight_decay_sda', type=float, default=1e-6, help="Weight decay for SDA")
    
    # Loss function parameters
    parser.add_argument('--wd_clf', type=int, default=1, help="wasserstein dist factor")
    parser.add_argument('--clip_value', type=float, default=0.01, help="clip gradients")
    parser.add_argument('--gamma', type=float, default=20, help="gamma in gradient penalty")
    parser.add_argument('--alpha', type=float, default=0.75, help="alpha in SDA loss")
    parser.add_argument('--margin', type=float, default=15, help="margin in SDA loss")
    
    # Miscellaneous
    parser.add_argument('--iter', type=str, default='none', help="iteration")
    parser.add_argument('--intgrade', type=bool, default=False, help="integrated gradients")
    parser.add_argument('--train_source', action='store_true', help="Train source model")
    parser.add_argument('--adapt', action='store_true', help="Perform adaptation")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate model")
    parser.add_argument('--save_losses', action='store_true', help="Whether to save losses")
    parser.add_argument('--desired_fpr', type=float, default=0.1, help="Desired FPR for evaluation")
    
    # Dataset sizes
    parser.add_argument('--src_size', type=int, default=30000, help="Number of positive examples in source dataset")
    parser.add_argument('--src_neg_size', type=int, default=None, help="Number of negative examples in source dataset (equal to src_size if not provided)")
    parser.add_argument('--tgt_size', type=int, default=1754, help="Number of positive examples in target dataset")
    parser.add_argument('--tgt_neg_size', type=int, default=None, help="Number of negative examples in target dataset (equal to tgt_size if not provided)")
    parser.add_argument('--test_pos_size', type=int, default=200, help="Number of positive examples in test dataset")
    parser.add_argument('--test_neg_size', type=int, default=20000, help="Number of negative examples in test dataset")
    
    # Batch size and random seed
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")
    
    # SDA benchmark
    parser.add_argument('--sda_benchmark', action='store_true', help="Use SDA benchmark configuration for source dataset")
    
    # Spiral galaxies configuration
    parser.add_argument('--include_spirals', action='store_true', help="Whether to include spiral galaxies in the datasets")
    parser.add_argument('--total_spirals', type=int, default=1000, help="Total number of spiral galaxies for training")
    parser.add_argument('--spiral_source_fraction', type=float, default=0.7, help="Fraction of training spirals to use in source")
    parser.add_argument('--test_spirals', type=int, default=200, help="Number of spiral galaxies in test set")



    return parser.parse_args()

def setup_logger(out_dir):
    """
    Set up logger to write to both file and console
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_file = os.path.join(out_dir, f'execution_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging to: {log_file}")
    return logging.getLogger()

def print_dataset_filenames(loader, dataset_name, out_dir):
    """
    Print all filenames from a given loader to a separate file
    
    Args:
        loader: DataLoader instance
        dataset_name: name of dataset for logging
        out_dir: output directory path
    """
    # Collect all IDs
    logging.info(f"Start writing filenames for {dataset_name}")
    all_ids = []
    for _, _, ids in loader:
        all_ids.extend(ids)
    all_ids = sorted(all_ids)
    
    # Write to separate file
    filename = os.path.join(out_dir, f'{dataset_name}_filenames.txt')
    with open(filename, 'w') as f:
        f.write(f"Filenames in {dataset_name}:\n")
        for file_id in all_ids:
            f.write(f"{file_id}\n")
    
    # Log only the summary
    logging.info(f"Filenames for {dataset_name} written to: {filename}")
    logging.info(f"Total files in {dataset_name}: {len(all_ids)}")
    return 0

def main():
    print("Start main")
    args = parse_arguments()
    torch.manual_seed(30)  # 128
    torch.set_default_dtype(torch.float32)
    np.set_printoptions(precision=16)  
    
    # Setup logger
    setup_logger(args.out_dir)
    
    # Log basic information
    logging.info(f"Starting execution with arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Setup preprocessing
    func_dict = {"none": idnorm, "sqrt": sqrtnorm, "log": lognorm}
    preprocess = func_dict[args.preprocess]
    
    
    # Initialize DataReader
    data_reader = DataReader(args.data_dir, preprocess)
    
    # Only the 'test' split is needed when neither training the source model
    # nor running adaptation (i.e. a pure evaluation run on existing weights)
    only_test = not (args.train_source or args.adapt)

    # Configure data loading
    data_config = {
        'source_size': (args.src_size, args.src_neg_size or args.src_size),
        'target_size': (args.tgt_size, args.tgt_neg_size or args.tgt_size),
        'test_size': (args.test_pos_size, args.test_neg_size),
        'batch_size': args.batch_size,
        'random_seed': args.random_seed,
        'transforms': image_transforms,
        'sda_benchmark': args.sda_benchmark,
        'only_test': only_test,
        'spirals_config': {
            'use_spirals': args.include_spirals,
            'train_spirals_size': args.total_spirals,
            'source_fraction': args.spiral_source_fraction,
            'test_spirals_size': args.test_spirals
        }
    }

    # Get all dataloaders
    # loaders = data_reader.get_all_loaders(data_config)
    loaders = data_reader.get_default_loaders(data_config)

    # Print filenames for target_train and test loaders
    # print_dataset_filenames(loaders['target_train'], "target_train", args.out_dir)
    # print_dataset_filenames(loaders['test'], "test", args.out_dir)

    source_train_loader = loaders.get('source_train')
    source_val_loader = loaders.get('source_val')
    target_train_loader = loaders.get('target_train')
    target_val_loader = loaders.get('target_val')
    test_loader = loaders['test']

    config = TrainingConfig(
        # General training parameters
        patience=args.patience,
        min_delta=args.min_delta,
        save_interval=args.save_interval,
        save_losses=args.save_losses,

        # Encoder and method-specific epochs
        n_epochs=args.n_epochs_enc,
        adda_epochs=args.adda_epochs,
        wdgrl_epochs=args.wdgrl_epochs,
        sda_epochs=args.sda_epochs,

        # Learning rates from command line arguments
        lr_source=args.lr_source,
        lr_source_scheduler=args.lr_source_scheduler,
        lr_adda_critic=args.lr_adda_critic,
        lr_adda_enc=args.lr_adda_enc,
        lr_wdgrl_critic=args.lr_wdgrl_critic,
        lr_wdgrl_clf=args.lr_wdgrl_clf,
        lr_wdgrl_encoder=args.lr_wdgrl_encoder,
        lr_sda=args.lr_sda,

        # Weight decay parameters from command line arguments
        weight_decay_source=args.weight_decay_source,
        weight_decay_adda_critic=args.weight_decay_adda_critic,
        weight_decay_adda_enc=args.weight_decay_adda_enc,
        weight_decay_wdgrl_critic=args.weight_decay_wdgrl_critic,
        weight_decay_wdgrl_clf=args.weight_decay_wdgrl_clf,
        weight_decay_wdgrl_encoder=args.weight_decay_wdgrl_encoder,
        weight_decay_sda=args.weight_decay_sda,

        # Method-specific parameters
        adda_n_encoder_steps=args.adda_n_encoder_steps,
        n_critic=args.wdgrl_n_critic_steps,
        n_clf=args.wdgrl_n_clf_steps,
        wd_clf=args.wd_clf,
        gamma=args.gamma,
        margin=args.margin,
        alpha=args.alpha,

        # Model and output configuration
        encoder_name=args.encoder,
        DA_methods=args.DA_methods.split('/') if args.DA_methods != "all" else ["adda", "wdgrl", "sda"],
        iteration=args.iter if args.iter != 'none' else 0,
        output_dir=args.out_dir,
        savename=args.savename
    )

    logging.info(f"Configuration: {config}")

    # Initialize model
    model = CombinedModel(config)
    # print(model.source_encoder.block1.state_dict()['1.weights'])
    # print(model.source_classifier.fc2.weight[:, :10])
    
    # Load or train source model
    if args.train_source:
        logging.info("Training source encoder...")
        model.train_backbone(source_train_loader, source_val_loader)
    else:
        try:
            model.load_source_weights()
            logging.info("Loaded source weights.")
        except FileNotFoundError as e:
            logging.error(f"Error loading source weights: {e}")
            return

    # Perform adaptation or load adapted weights
    if args.adapt:
        logging.info("Running domain adaptation...")
        model.run_adaptation(source_train_loader, target_train_loader, target_val_loader)
        loaded_methods = config.DA_methods
    else:
        try:
            loaded_methods = model.load_target_weights()
            logging.info("Loaded target weights.")
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"Error loading target weights: {e}")
            logging.warning("Proceeding with evaluation of naive method only")
            loaded_methods = []

    # Evaluate model
    if args.evaluate:
        evaluation_results = model.evaluate(test_loader, loaded_methods, desired_fpr=args.desired_fpr)

        # Plot ROC curves and Precision-Recall curves
        output_dir = os.path.join(args.out_dir, f'{args.encoder}')
        os.makedirs(output_dir, exist_ok=True)

        plot_roc_curves(evaluation_results, args, output_dir)
        plot_roc_curves_log(evaluation_results, args, output_dir)
        plot_precision_recall_curves(evaluation_results, args, output_dir)

        logging.info("Evaluation and plotting completed.")

if __name__ == "__main__":
    main()

from util_module import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split # type: ignore
import numpy as np
import os
import glob
from torchvision import transforms # type: ignore
import logging
from preprocess_module import makeimg, preprocess


image_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5), 
        transforms.RandomRotation([-90, +90]), 
        transforms.CenterCrop(64),  
    ]),

    "val": transforms.Compose([
        transforms.CenterCrop(64)
    ])
}


class DirData(Dataset):
    def __init__(self, fnames, labels, transform, preprocess_func):
        self.fnames = fnames
        self.labels = labels
        self.transform = transform
        self.preprocess_func = preprocess_func

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, i):
        img_path = self.fnames[i]
        idforTest = self.extract_id(img_path)
        
        # Load and preprocess image
        img = makeimg(img_path)  # Convert to torch tensor with proper shape
        if self.transform:
            # print("transform", self.transform)
            img = self.transform(img)
        img = preprocess(img)  # Normalize and handle NaN values
        # print("After preprocess", img.max(), img.min(), img.mean(), img.std())
        
        if self.preprocess_func:
            # print("preprocess func", self.preprocess_func)
            img = self.preprocess_func(img)
        # print("After preprocess", img.max(), img.min(), img.mean(), img.std())
        # if img.isnan().any():
        #     print(img_path, "have nan")
        
        return img, self.labels[i], idforTest

    def extract_id(self, img_path):
        """Extract folder and filename from path"""
        # return os.path.basename(img_path).split('.')[0]
        return "/".join(img_path.split("/")[-2:])


class DataReader:
    def __init__(self, data_dir, preprocess_func):
        self.data_dir = data_dir
        self.preprocess_func = preprocess_func
        self.rng = np.random.default_rng(42)

    def get_all_loaders(self, config):
        """
        Returns all necessary dataloaders based on configuration
        """
        # Set random seed for reproducibility
        rng = np.random.default_rng(config['random_seed'])
        
        # Load all files
        source_pos = self._load_positive_files('source')
        real_pos = self._load_positive_files('target')
        source_neg = self._load_negative_files('source')
        target_neg = self._load_negative_files('target')
        test_neg = self._load_negative_files('test')
        
        # Log initial file counts
        logging.info("\nInitial file counts:")
        logging.info(f"Source positives: {len(source_pos)}")
        logging.info(f"Real positives (total): {len(real_pos)}")
        logging.info(f"Source negatives: {len(source_neg)}")
        logging.info(f"Target negatives: {len(target_neg)}")
        logging.info(f"Test negatives: {len(test_neg)}")
        
        # Shuffle all file arrays
        #for files in [source_pos, real_pos, source_neg, target_neg, test_neg]:
        #    rng.shuffle(files)

        # experiment_val_pos = self._load_positive_files('experiment_val')
        # experiment_val_neg = source_neg[-len(experiment_val_pos):]
        # experiment_val_dataset = ConcatDataset([
        #     DirData(experiment_val_pos, np.ones(len(experiment_val_pos)), config['transforms']['val'], self.preprocess_func),
        #     DirData(experiment_val_neg, np.zeros(len(experiment_val_neg)), config['transforms']['val'], self.preprocess_func)
        # ])
        
        # Split real positives between target and test
        test_pos, target_pos = self._split_files(real_pos, test_fraction=0.1)

        def adjust_size(files, requested_size, dataset_name):
            """
            Adjust array size according to requested size
            
            Args:
                files: array of file paths
                requested_size: desired number of files
                dataset_name: name of dataset for logging
            
            Returns:
                Array trimmed to requested size, or full array if requested size exceeds available files
            """
            available_size = len(files)
            if requested_size > available_size:
                logging.warning(
                    f"Requested size for {dataset_name} ({requested_size}) exceeds "
                    f"available files ({available_size}). Using all available files."
                )
                return files
            return files[:requested_size]

        # Adjust sizes according to config
        source_pos = adjust_size(source_pos, config['source_size'][0], "source positives")
        target_pos = adjust_size(target_pos, config['target_size'][0], "target positives")
        test_pos = adjust_size(test_pos, config['test_size'][0], "test positives")
        
        source_neg = adjust_size(source_neg, config['source_size'][1], "source negatives")
        target_neg = adjust_size(target_neg, config['target_size'][1], "target negatives")
        test_neg = adjust_size(test_neg, config['test_size'][1], "test negatives")

        # Handle spirals if enabled 
        if config['spirals_config']['use_spirals']:
            spiral_files = self._load_spiral_files()
            logging.info(f"\nTotal spiral galaxies available: {len(spiral_files)}")
            
            rng.shuffle(spiral_files)
            total_spiral_files = len(spiral_files)
            requested_total = (config['spirals_config']['test_spirals_size'] + 
                              config['spirals_config']['train_spirals_size'])
            
            if requested_total > total_spiral_files:
                logging.warning(
                    f"Requested number of spiral galaxies ({requested_total}) exceeds "
                    f"available files ({total_spiral_files}). Using all available files."
                )
                ratio = config['spirals_config']['train_spirals_size'] / requested_total
                train_size = int(total_spiral_files * ratio)
                test_size = total_spiral_files - train_size
            else:
                train_size = config['spirals_config']['train_spirals_size']
                test_size = config['spirals_config']['test_spirals_size']
            
            # Split files
            test_spirals = spiral_files[-test_size:]
            train_spirals = spiral_files[:train_size]
            
            # Split train spirals between source/target
            source_spiral_size = int(len(train_spirals) * config['spirals_config']['source_fraction'])
            source_spirals = train_spirals[:source_spiral_size]
            target_spirals = train_spirals[source_spiral_size:]
            
            # Add spirals to negative sets
            source_neg = np.concatenate([source_neg, source_spirals])
            target_neg = np.concatenate([target_neg, target_spirals])
            test_neg = np.concatenate([test_neg, test_spirals])
            
            # Log spiral distribution
            logging.info("\nSpiral galaxies distribution:")
            logging.info(f"Source: {len(source_spirals)}")
            logging.info(f"Target: {len(target_spirals)}")
            logging.info(f"Test: {len(test_spirals)}")

        # Handle SDA benchmark case after all adjustments
        if config.get('sda_benchmark', False):
            logging.info("\nUsing SDA benchmark configuration:")
            logging.info("Training naive model purely on unbalanced real data")
            logging.info("Using target positives as source positives")
            logging.info("Combining source and target negatives for source dataset")
            
            # Use target positives as source positives
            source_pos = target_pos.copy()
            # Combine negatives for source (now including any spiral galaxies)
            combined_neg = np.concatenate([source_neg, target_neg])
            rng.shuffle(combined_neg)  # Shuffle combined negatives
            # source_neg = combined_neg[:config['source_size'][1]]  # Take required number of negatives
            
            logging.info(f"Combined source negatives size: {len(source_neg)}")

        # Create datasets
        source_dataset = ConcatDataset([
            DirData(source_pos, np.ones(len(source_pos)), config['transforms']['train'], self.preprocess_func),
            DirData(source_neg, np.zeros(len(source_neg)), config['transforms']['train'], self.preprocess_func)
        ])
        
        target_dataset = ConcatDataset([
            DirData(target_pos, np.ones(len(target_pos)), config['transforms']['train'], self.preprocess_func),
            DirData(target_neg, np.zeros(len(target_neg)), config['transforms']['train'], self.preprocess_func)
        ])
        
        test_dataset = ConcatDataset([
            DirData(test_pos, np.ones(len(test_pos)), config['transforms']['val'], self.preprocess_func),
            DirData(test_neg, np.zeros(len(test_neg)), config['transforms']['val'], self.preprocess_func)
        ])

        # Split source and target into train/val
        source_train, source_val = random_split(
            source_dataset,
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(config['random_seed'])
        )
        
        target_train, target_val = random_split(
            target_dataset,
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(config['random_seed'])
        )

        # source_val = experiment_val_dataset

        # Log final dataset sizes
        logging.info("\nFinal dataset sizes:")
        logging.info(f"Source - Train: {len(source_train)}, Val: {len(source_val)}")
        logging.info(f"Target - Train: {len(target_train)}, Val: {len(target_val)}")
        logging.info(f"Test: {len(test_dataset)}")
        
        # Create dataloaders
        loaders = {
            'source_train': DataLoader(source_train, batch_size=config['batch_size'], shuffle=False, drop_last=True),
            'source_val': DataLoader(source_val, batch_size=config['batch_size'], shuffle=False),
            'target_train': DataLoader(target_train, batch_size=config['batch_size'], shuffle=False, drop_last=True),
            'target_val': DataLoader(target_val, batch_size=config['batch_size'], shuffle=False),
            'test': DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        }

        return loaders

    def _load_positive_files(self, dataset_type):
        """
        Load positive examples (lenses) based on dataset type
        
        Args:
            dataset_type: 'source', 'target', or 'test'
        """
        if dataset_type == 'source':
            path = os.path.join(self.data_dir, 'holismokes', 'hsc_hst_holismokes_npy')
        elif dataset_type == 'experiment_val':
            path = os.path.join(self.data_dir, 'holismokes', 'real_lenses', 'clean_real_lenses', 'experiment_val_real_lenses_npy')
        else:  # test
            path = os.path.join(self.data_dir, 'holismokes', 'real_lenses', 'clean_real_lenses', 'experiment_real_lenses_npy')
        
        return glob.glob(os.path.join(path, '*.npy'))

    def _load_negative_files(self, dataset_type):
        """Load negative examples (non-lenses) for specific dataset"""
        if dataset_type == 'source':
            path = os.path.join(self.data_dir, 'holismokes', 'nonlenses', 'nonlenses4train_npy')
        elif dataset_type == 'target':
            path = os.path.join(self.data_dir, 'holismokes', 'nonlenses', 'nonlenses4adapt_npy')
        else:  # test
            path = os.path.join(self.data_dir, 'holismokes', 'nonlenses', 'nonlenses4test_npy')
            
        files = glob.glob(os.path.join(path, '*.npy'))
        self.rng.shuffle(files)  # Shuffle files in place
        return files

    def _load_spiral_files(self):
        """Load all spiral galaxy files"""
        path = os.path.join(self.data_dir, 'holismokes', 'nonlenses', 'tadaki_full_npy')
        files = glob.glob(os.path.join(path, '*.npy'))
        self.rng.shuffle(files)
        return files

    def _split_files(self, files, test_fraction):
        """Split files into two parts with given fraction"""
        split_idx = int(len(files) * (1 - test_fraction))
        return files[split_idx:], files[:split_idx]

    def get_default_loaders(self, config):
        """
        Returns dataloaders by loading pre-split datasets from specified paths.

        If config['only_test'] is True, skips loading the source/target
        train+val splits and the simulated test split entirely, and returns
        just the 'test' loader (those other splits aren't needed to run
        evaluation on pre-trained weights).
        """
        only_test = config.get('only_test', False)

        loaders = {}

        if not only_test:
            # Load training files
            source_pos_train = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/train_source_lenses'))
            source_neg_train = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/train_source_nonlenses'))
            target_pos_train = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/train_target_lenses'))
            target_neg_train = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/train_target_nonlenses'))

            # Load validation files
            source_pos_val = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/val_source_lenses_fair'))
            source_neg_val = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/val_source_nonlenses_fair'))
            target_pos_val = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/val_target_lenses'))
            target_neg_val = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/val_target_nonlenses'))

            test_sim_pos = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/test_sim_lenses'))
            test_sim_neg = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/test_sim_nonlenses'))

            logging.info("\nDataset sizes:")
            logging.info(f"Source train - Pos: {len(source_pos_train)}, Neg: {len(source_neg_train)}")
            logging.info(f"Source val - Pos: {len(source_pos_val)}, Neg: {len(source_neg_val)}")
            logging.info(f"Target train - Pos: {len(target_pos_train)}, Neg: {len(target_neg_train)}")
            logging.info(f"Target val - Pos: {len(target_pos_val)}, Neg: {len(target_neg_val)}")
            logging.info(f"Test sim - Pos: {len(test_sim_pos)}, Neg: {len(test_sim_neg)}")

            source_train = ConcatDataset([
                DirData(source_pos_train, np.ones(len(source_pos_train)), config['transforms']['train'], self.preprocess_func),
                DirData(source_neg_train, np.zeros(len(source_neg_train)), config['transforms']['train'], self.preprocess_func)
            ])

            source_val = ConcatDataset([
                DirData(source_pos_val, np.ones(len(source_pos_val)), config['transforms']['val'], self.preprocess_func),
                DirData(source_neg_val, np.zeros(len(source_neg_val)), config['transforms']['val'], self.preprocess_func)
            ])

            target_train = ConcatDataset([
                DirData(target_pos_train, np.ones(len(target_pos_train)), config['transforms']['train'], self.preprocess_func),
                DirData(target_neg_train, np.zeros(len(target_neg_train)), config['transforms']['train'], self.preprocess_func)
            ])

            target_val = ConcatDataset([
                DirData(target_pos_val, np.ones(len(target_pos_val)), config['transforms']['val'], self.preprocess_func),
                DirData(target_neg_val, np.zeros(len(target_neg_val)), config['transforms']['val'], self.preprocess_func)
            ])

            test_sim_dataset = ConcatDataset([
                DirData(test_sim_pos, np.ones(len(test_sim_pos)), config['transforms']['val'], self.preprocess_func),
                DirData(test_sim_neg, np.zeros(len(test_sim_neg)), config['transforms']['val'], self.preprocess_func)
            ])

            loaders['source_train'] = DataLoader(source_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
            loaders['source_val'] = DataLoader(source_val, batch_size=config['batch_size'], shuffle=True, drop_last=True)
            loaders['target_train'] = DataLoader(target_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
            loaders['target_val'] = DataLoader(target_val, batch_size=config['batch_size'], shuffle=True, drop_last=True)
            loaders['test_sim'] = DataLoader(test_sim_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

        # Load test files (always needed)
        test_pos = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/test_real_lenses'))
        test_neg = self._load_files(os.path.join(self.data_dir, 'holismokes/reproducing_default/test_real_nonlenses'))

        logging.info(f"Test - Pos: {len(test_pos)}, Neg: {len(test_neg)}")

        test_dataset = ConcatDataset([
            DirData(test_pos, np.ones(len(test_pos)), config['transforms']['val'], self.preprocess_func),
            DirData(test_neg, np.zeros(len(test_neg)), config['transforms']['val'], self.preprocess_func)
        ])

        print("len test dataset", len(test_pos) + len(test_neg))

        loaders['test'] = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

        return loaders

    def _load_files(self, path):
        """Helper function to load files from a directory"""
        # print(sorted(glob.glob(os.path.join(path, '*.npy')))[:4])
        return sorted(glob.glob(os.path.join(path, '*.npy')))

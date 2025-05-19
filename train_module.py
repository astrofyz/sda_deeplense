from util_module import *
from model import *
import logging
import datetime
import os


class EarlyStopping:  # need to be updated
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def early_stop(self, val_loss):
        print("Early stopping, ", self.best_loss, val_loss)
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TrainingConfig(NamedTuple):
    # General training parameters
    patience: int = 5
    min_delta: float = 0.0001
    save_interval: int = 5
    save_losses: bool = False

    # Encoder and method-specific epochs
    n_epochs: int = 20
    adda_epochs: int = 20
    wdgrl_epochs: int = 20
    sda_epochs: int = 20

    # Learning rates
    lr_source: float = 1e-5
    lr_source_scheduler: float = 0.002
    lr_adda_critic: float = 1e-5
    lr_adda_enc: float = 1e-6
    lr_wdgrl_critic: float = 1e-4
    lr_wdgrl_clf: float = 1e-4 
    lr_wdgrl_encoder: float = 1e-4
    lr_sda: float = 1e-3

    # Weight decay parameters
    weight_decay_source: float = 1e-5
    weight_decay_adda_critic: float = 2.5e-4
    weight_decay_adda_enc: float = 2.5e-5
    weight_decay_wdgrl_critic: float = 1e-6
    weight_decay_wdgrl_clf: float = 1e-6
    weight_decay_wdgrl_encoder: float = 1e-6
    weight_decay_sda: float = 1e-6

    # Method-specific parameters
    adda_n_encoder_steps: int = 3
    n_critic: int = 5
    n_clf: int = 1
    wd_clf: float = 1.0
    gamma: float = 10.0
    margin: float = 1.0
    alpha: float = 0.1

    # Model and output configuration
    encoder_name: str = "ENN"
    DA_methods: list = []
    iteration: str = "none"
    output_dir: str = "ENN"
    savename: str = ""


def setup_training_logger(output_dir):
    # Create a new logger for training
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logs
    training_logger.handlers = []
    
    # Create handlers
    log_file = os.path.join(output_dir, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    training_logger.addHandler(file_handler)
    training_logger.addHandler(console_handler)
    
    return training_logger


class CombinedModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super(CombinedModel, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.source_encoder = encoder_dict[config.encoder_name]().to(self.device)
        self.source_classifier = Classifier().to(self.device)
        
        self.target_encoder = dict()
        for method in config.DA_methods:
            self.target_encoder[method] = encoder_dict[config.encoder_name]().to(self.device)
        
        self.target_classifier = dict()
        for method in config.DA_methods:
            self.target_classifier[method] = Classifier().to(self.device)
            
        self.logger = setup_training_logger(config.output_dir)
        self.logger.info(f"CombinedModel initialized with {self.config.encoder_name} encoder and methods: {self.config.DA_methods}")

    def forward(self, x, domain='source', method='adda'):
        if domain == 'source':
            # print("Encoder data:")
            # print(self.source_encoder(x).cpu().detach().numpy())
            return self.source_classifier(self.source_encoder(x))
        elif domain == 'target':
            return self.target_classifier[method](self.target_encoder[method](x))
        else:
            raise ValueError("Domain must be 'source' or 'target'")

    def train_backbone(self, train_loader, val_loader, continue_training=False):
        if continue_training:
            self.logger.info("loading weights")
            self.source_encoder.load_state_dict(torch.load(os.path.join(self.config.output_dir, f'{self.config.encoder_name}', f'{self.config.encoder_name}_Encoder_Source_{self.config.iteration}.pth')))
            self.source_classifier = torch.load(os.path.join(self.config.output_dir, f'{self.config.encoder_name}', f'{self.config.encoder_name}_Classifier_Source_{self.config.iteration}.pth'))

        # Print model structure
#        self.logger.info("\nSource Encoder Structure:")
#        self.logger.info(self.source_encoder)
#        self.logger.info("\nSource Classifier Structure:")
#        self.logger.info(self.source_classifier)

        optimizer = torch.optim.Adam(
            list(self.source_encoder.parameters()) + list(self.source_classifier.parameters()),
            lr=self.config.lr_source,
            weight_decay=self.config.weight_decay_source
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.config.lr_source_scheduler,
            epochs=self.config.n_epochs,
            steps_per_epoch=len(train_loader)
        )
        criteria = nn.CrossEntropyLoss()

        early_stopper = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta)

        losses = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        

        for epoch in range(1, self.config.n_epochs + 1):
            # print("Weights before training:")
            # print(self.source_encoder.block1.state_dict()['1.weights'])
            # print(self.source_classifier.fc2.weight)
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler, criteria)
            # print("Weights after training:")
            # print(self.source_encoder.block1.state_dict()['1.weights'])
            # print(self.source_classifier.fc2.weight)
            val_loss, val_acc = self._validate(val_loader, criteria)

            self.logger.info(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f}')
            self.logger.info(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f} \tValidation Acc: {val_acc:.6f}')
            self.logger.info('')

            if self.config.save_losses:
                losses['train_loss'].append(train_loss)
                losses['train_acc'].append(train_acc)
                losses['val_loss'].append(val_loss)
                losses['val_acc'].append(val_acc)

            self._save_models(method='source')

            # Print optimizer state at the end of each epoch
            self.logger.info(f"Source encoder optimizer state:")
            for group in optimizer.param_groups:
                self.logger.info(f"Learning rate: {group['lr']}")
                self.logger.info(f"Weight decay: {group['weight_decay']}")

            if early_stopper.early_stop(val_loss):
                print(f"early stopping after Epoch {epoch}")
                break

            if self.config.save_losses:
                self._save_losses(losses, method='source')

        self._save_models(method='source')

    def _train_epoch(self, train_loader, optimizer, scheduler, criteria):
        self.source_encoder.train()
        self.source_classifier.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch in train_loader:
            data, labels = batch[0].float().to(self.device), batch[1].type(torch.LongTensor).reshape((-1,)).to(self.device)
            
            # print("Data:")
            # print(batch[0].float()[np.where(batch[0].float() != 0.)])
            # print(batch[1].type(torch.LongTensor).reshape((-1,)))


            optimizer.zero_grad()
            outputs = self.forward(data, domain='source')
            _, preds = torch.max(outputs.data, 1)
            correct = (preds == labels).float().sum()
            loss = criteria(outputs.float(), labels)
            # print("Weights:")
            # print(self.source_encoder.block1.state_dict()['1.weights'])
            # print(self.source_classifier.fc2.weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print("Weights after update:")
            # print(self.source_encoder.block1.state_dict()['1.weights'])
            # print(self.source_classifier.fc2.weight)
            
            # print("Next batch\n")  

                 
            train_loss += loss.item()
            train_acc += correct.item() / data.shape[0]

        return train_loss / len(train_loader), train_acc / len(train_loader)

    def _validate(self, val_loader, criteria):
        self.source_encoder.eval()
        self.source_classifier.eval()
        val_loss = 0.
        val_acc = 0.

        with torch.no_grad():
            for batch in val_loader:
                data, labels = batch[0].float().to(self.device), batch[1].type(torch.LongTensor).reshape((-1,)).to(self.device)
                # print("Validation Data:")
                # print(batch[0].float().detach().cpu().numpy().shape, batch[0].float().detach().cpu().numpy()[0, 0, :, :], batch[0].float().detach().cpu().numpy().std())
                # print(batch[1].type(torch.LongTensor).reshape((-1,)))

                outputs = self.forward(data, domain='source')
                # print("Validation Weights:")
                # print(self.source_encoder.block1.state_dict()['1.weights'])
                # print(self.source_encoder.block1.state_dict()['2.weights'])
                # print(self.source_encoder.block1.state_dict()['3.weights'])
                # print(self.source_encoder.block2.state_dict()['1.weights'])
                # print(self.source_encoder.block2.state_dict()['2.weights'])
                # print(self.source_encoder.block3.state_dict()['1.weights'])
                # print(self.source_encoder.block3.state_dict()['2.weights'])
                # print(self.source_classifier.fc2.weight)
                # print(outputs.data)
                _, preds = torch.max(outputs.data, 1)
                correct = (preds == labels).float().sum()
                loss = criteria(outputs, labels)

                # print("Validation Next batch\n")    
                 

                val_loss += loss.item()
                val_acc += correct.item() / data.shape[0]

        return val_loss / len(val_loader), val_acc / len(val_loader)

    def _save_models(self, method='source', epoch=None):
        model_dir = os.path.join(self.config.output_dir, f'{self.config.encoder_name}', 'weights', method)
        os.makedirs(model_dir, exist_ok=True)
        
        if method == 'source':
            encoder = self.source_encoder
            classifier = self.source_classifier
        else:
            encoder = self.target_encoder[method]
            classifier = self.target_classifier[method]
        
        epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
        iteration_str = f'_{self.config.iteration}' if hasattr(self.config, 'iteration') else ''
        
        torch.save(encoder.state_dict(), 
                   os.path.join(model_dir, f'{self.config.encoder_name}_encoder_{method}{epoch_str}{iteration_str}.pth'))
        torch.save(classifier.state_dict(), 
                   os.path.join(model_dir, f'{self.config.encoder_name}_classifier_{method}{epoch_str}{iteration_str}.pth'))

    def _save_losses(self, losses, method='source'):
        losses_dir = os.path.join(self.config.output_dir, f'{self.config.encoder_name}', 'losses')
        os.makedirs(losses_dir, exist_ok=True)
        iteration_str = f'_{self.config.iteration}' if hasattr(self.config, 'iteration') else ''
        losses_file = os.path.join(losses_dir, f'{self.config.encoder_name}_losses_{method}{iteration_str}.pkl')
        with open(losses_file, 'wb') as f:
            pickle.dump(losses, f)

    def _adapt_adda(self, source_loader, target_loader, target_val_loader):
        device = self.device
        early_stopper = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta)

        # Create a new instance of the encoder for the target
        self.target_encoder['adda'] = encoder_dict[self.config.encoder_name]().to(device)
        # Load the state dict of the source encoder into the target encoder
        self.target_encoder['adda'].load_state_dict(self.source_encoder.state_dict())
        self.target_encoder['adda'].train()

        self.target_classifier['adda'] = Classifier().to(device)
        # Load the state dict of the source encoder into the target encoder
        self.target_classifier['adda'].load_state_dict(self.source_classifier.state_dict())
        self.target_classifier['adda'].eval()
        
        # Ensure the source encoder is in eval mode and its weights are frozen
        self.source_encoder.eval()
        self.source_classifier.eval()
        for param in self.source_encoder.parameters():
            param.requires_grad = False
        for param in self.source_classifier.parameters():
            param.requires_grad = False

        critic = Discriminator().to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer_tgt = optim.Adam(self.target_encoder['adda'].parameters(), 
                                 lr=self.config.lr_adda_enc, 
                                 betas=(0.5, 0.999), 
                                 weight_decay=self.config.weight_decay_adda_enc)
        optimizer_critic = optim.Adam(critic.parameters(), 
                                    lr=self.config.lr_adda_critic, 
                                    betas=(0.5, 0.999), 
                                    weight_decay=self.config.weight_decay_adda_critic)

        losses = {'critic_loss': [], 'encoder_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, self.config.adda_epochs + 1):
            # self.logger.info(f'\nEpoch {epoch} - Sample weights comparison:')

            critic_loss_cumul = 0.
            encoder_loss_cumul = 0.
            
            critic.train()
            self.target_encoder['adda'].train()

            for batch_src, batch_tgt in zip(source_loader, target_loader):
                images_src, labels_src = batch_src[0].to(device).float(), batch_src[1].to(device).long().reshape((-1,))
                images_tgt = batch_tgt[0].to(device).float()

                # Train Discriminator
                optimizer_critic.zero_grad()
                with torch.no_grad():
                    D_input_source = self.source_encoder(images_src)
                    D_input_target = self.target_encoder['adda'](images_tgt)
                D_output_source = critic(D_input_source.detach())
                D_output_target = critic(D_input_target.detach())
                D_target_source = torch.zeros(images_src.size(0), dtype=torch.long).to(device)
                D_target_target = torch.ones(images_tgt.size(0), dtype=torch.long).to(device)
                D_output = torch.cat([D_output_source, D_output_target], dim=0)
                D_target = torch.cat([D_target_source, D_target_target], dim=0)
                d_loss = criterion(D_output, D_target)
                d_loss.backward()
                optimizer_critic.step()

                # Train Target Encoder
                for _ in range(self.config.adda_n_encoder_steps):
                    optimizer_tgt.zero_grad()
                    D_input_target = self.target_encoder['adda'](images_tgt)
                    D_output_target = critic(D_input_target)
                    D_target_source = torch.zeros(images_tgt.size(0), dtype=torch.long).to(device)
                    loss = criterion(D_output_target, D_target_source)


                    loss.backward()
                    optimizer_tgt.step()
                    #might need to add a scheduler here

                    encoder_loss_cumul += loss.item()
                
                critic_loss_cumul += d_loss.item()


            self.logger.info(f"D_input_target requires grad: {D_input_target.requires_grad}")
            self.logger.info(f"D_output_target requires grad: {D_output_target.requires_grad}")


            # Validation
            val_loss, val_acc = self._validate_adaptation(self.target_encoder['adda'], target_val_loader)
            
            losses['critic_loss'].append(critic_loss_cumul / len(target_loader))
            losses['encoder_loss'].append(encoder_loss_cumul / len(target_loader) / self.config.adda_n_encoder_steps)
            losses['val_loss'].append(val_loss)
            losses['val_acc'].append(val_acc)
            
            self.logger.info(f'Epoch: {epoch} \tCritic Loss: {critic_loss_cumul/len(target_loader):.6f} \tTgt Loss: {encoder_loss_cumul/len(target_loader)/self.config.adda_n_encoder_steps:.6f}')
            self.logger.info(f'Validation Loss: {val_loss:.6f} \tValidation Accuracy: {val_acc:.6f}')
            
            if epoch % self.config.save_interval == 0:
                self._save_models(method='adda', epoch=epoch)
            
            # if early_stopper.early_stop(val_loss):
            #     print(f"Early stopping after Epoch {epoch}")
            #     self._save_models(method='adda', epoch=epoch)
            #     break

            # At the end of each epoch, print optimizer state
            self.logger.info(f"Target encoder optimizer state:")
            for group in optimizer_tgt.param_groups:
                self.logger.info(f"Learning rate: {group['lr']}")
                self.logger.info(f"Weight decay: {group['weight_decay']}")

        self._save_losses(losses, method='adda')

    def _adapt_wdgrl(self, source_loader, target_loader, target_val_loader):
        device = self.device
        early_stopper = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta)

        # Create a new instance of the encoder for the target
        self.target_encoder['wdgrl'] = encoder_dict[self.config.encoder_name]().to(device)
        # Load the state dict of the source encoder into the target encoder
        self.target_encoder['wdgrl'].load_state_dict(self.source_encoder.state_dict())
        self.target_encoder['wdgrl'].train()

        self.target_classifier['wdgrl'] = Classifier().to(device)
        # Load the state dict of the source encoder into the target encoder
        self.target_classifier['wdgrl'].load_state_dict(self.source_classifier.state_dict())
        self.target_classifier['wdgrl'].train()
        

        clf_criterion = nn.CrossEntropyLoss()
        critic = Discriminator().to(device)
        critic.layer[4] = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid()).to(device)

        critic_optim = torch.optim.Adam(critic.parameters(), 
                                      lr=self.config.lr_wdgrl_critic, 
                                      weight_decay=self.config.weight_decay_wdgrl_critic)
        clf_optim = torch.optim.Adam(self.target_classifier['wdgrl'].parameters(), 
                                    lr=self.config.lr_wdgrl_clf, 
                                    weight_decay=self.config.weight_decay_wdgrl_clf)
        encoder_optim = torch.optim.Adam(self.target_encoder['wdgrl'].parameters(), 
                                       lr=self.config.lr_wdgrl_encoder, 
                                       weight_decay=self.config.weight_decay_wdgrl_encoder)
        # Add Wasserstein distance to losses dictionary
        losses = {'critic_loss': [], 'clf_loss': [], 'total_loss': [], 'val_loss': [], 'val_acc': [], 'wasserstein_distance': []}

        for epoch in range(1, self.config.wdgrl_epochs + 1):
            critic_loss_cumul = 0.
            total_loss_cumul = 0.
            clf_loss_cumul = 0.
            wasserstein_distance_cumul = 0.

            critic.train()
            self.target_classifier['wdgrl'].train()
            self.target_encoder['wdgrl'].train()

            for batch_src, batch_tgt in zip(source_loader, target_loader):
                images_src, labels_src = batch_src[0].to(device).float(), batch_src[1].to(device).long().reshape((-1,))
                images_tgt = batch_tgt[0].to(device).float()

                # Train critic
                set_require_grad(self.target_encoder['wdgrl'], requires_grad=False)
                set_require_grad(self.target_classifier['wdgrl'], requires_grad=False)
                set_require_grad(critic, requires_grad=True)
                for _ in range(self.config.n_critic):
                    critic_optim.zero_grad()
                    with torch.no_grad():
                        h_s = self.target_encoder['wdgrl'](images_src).to(device)
                        h_t = self.target_encoder['wdgrl'](images_tgt).to(device)
                    gp = gradient_penalty(critic, h_s, h_t)
                    critic_s = critic(h_s)
                    critic_t = critic(h_t)
                    wasserstein_distance = critic_s.mean() - critic_t.mean()
                    wasserstein_distance_cumul += wasserstein_distance.item()
                    critic_cost = -wasserstein_distance + self.config.gamma * gp
                    critic_cost.backward()
                    critic_optim.step()
                    critic_loss_cumul += critic_cost.item()

                # Train classifier and encoder
                set_require_grad(self.target_encoder['wdgrl'], requires_grad=True)
                set_require_grad(self.target_classifier['wdgrl'], requires_grad=True)
                set_require_grad(critic, requires_grad=False)
                for _ in range(self.config.n_clf):
                    encoder_optim.zero_grad()
                    clf_optim.zero_grad()
                    src_features = self.target_encoder['wdgrl'](images_src)
                    tgt_features = self.target_encoder['wdgrl'](images_tgt)
                    src_preds = self.target_classifier['wdgrl'](src_features)
                    clf_loss = clf_criterion(src_preds, labels_src)
                    wasserstein_distance = critic(src_features).mean() - critic(tgt_features).mean()
                    loss = clf_loss + self.config.wd_clf * wasserstein_distance
                    loss.backward()
                    encoder_optim.step()
                    clf_optim.step()
                    total_loss_cumul += loss.item()
                    clf_loss_cumul += clf_loss.item()

            self.logger.info(f"Critic optimizer state:")
            for group in critic_optim.param_groups:
                self.logger.info(f"Learning rate: {group['lr']}")
                self.logger.info(f"Weight decay: {group['weight_decay']}")

            self.logger.info(f"Encoder optimizer state:")
            for group in encoder_optim.param_groups:
                self.logger.info(f"Learning rate: {group['lr']}")
                self.logger.info(f"Weight decay: {group['weight_decay']}")

            self.logger.info(f"Classifier optimizer state:")
            for group in clf_optim.param_groups:
                self.logger.info(f"Learning rate: {group['lr']}")
                self.logger.info(f"Weight decay: {group['weight_decay']}")

            # Validation
            val_loss, val_acc = self._validate_adaptation(self.target_encoder['wdgrl'], target_val_loader)
            avg_wasserstein_distance = wasserstein_distance_cumul / (len(target_loader) * self.config.n_critic)
            
            
            # Save losses to the dictionary
            losses['critic_loss'].append(critic_loss_cumul/len(target_loader)/self.config.n_critic)
            losses['clf_loss'].append(clf_loss_cumul/len(target_loader)/self.config.n_clf)
            losses['total_loss'].append(total_loss_cumul/len(target_loader)/self.config.n_clf)
            losses['val_loss'].append(val_loss)
            losses['val_acc'].append(val_acc)
            losses['wasserstein_distance'].append(avg_wasserstein_distance)

            self.logger.info(f'Epoch: {epoch} \tCritic Loss: {losses["critic_loss"][-1]:.6f} \tClf loss: {losses["clf_loss"][-1]:.4f} \tTotal loss: {losses["total_loss"][-1]:.5f}')
            self.logger.info(f'Average Wasserstein Distance: {avg_wasserstein_distance:.6f}')
            self.logger.info(f'Validation Loss: {val_loss:.6f} \tValidation Accuracy: {val_acc:.6f}')
            
            if epoch % self.config.save_interval == 0:
                self._save_models(method='wdgrl', epoch=epoch)
            
            # if early_stopper.early_stop(val_loss):
            #     print(f"Early stopping after Epoch {epoch}")
            #     self._save_models(method='wdgrl', epoch=epoch)
            #     break

        self._save_losses(losses, method='wdgrl')

    def _adapt_sda(self, source_loader, target_loader, target_val_loader):
        device = self.device
        early_stopper = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta)
        
        # Create a new instance of the encoder for the target
        self.target_encoder['sda'] = encoder_dict[self.config.encoder_name]().to(device)
        # Load the state dict of the source encoder into the target encoder
        self.target_encoder['sda'].load_state_dict(self.source_encoder.state_dict())
        self.target_encoder['sda'].train()

        self.target_classifier['sda'] = Classifier().to(device)
        # Load the state dict of the source encoder into the target encoder
        self.target_classifier['sda'].load_state_dict(self.source_classifier.state_dict())
        self.target_classifier['sda'].train()
        
        ce_loss = nn.CrossEntropyLoss()
        optim_tgt = optim.Adam(
            list(self.target_classifier['sda'].parameters()) + list(self.target_encoder['sda'].parameters()),
            lr=self.config.lr_sda,
            weight_decay=self.config.weight_decay_sda
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optim_tgt, mode='min', min_lr=1e-6, factor=0.5)  # check this number and how lr is changing during training

        # Initialize losses dictionary
        losses = {'total_loss': [], 'csa_loss': [], 'ce_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, self.config.sda_epochs + 1):
            total_loss = 0.
            total_csa_loss = 0.
            total_ce_loss = 0.

            self.target_classifier['sda'].train()
            self.target_encoder['sda'].train()

            for batch_src, batch_tgt in zip(source_loader, target_loader):
                images_src, labels_src = batch_src[0].to(device).float(), batch_src[1].to(device).long().reshape((-1,))
                images_tgt, labels_tgt = batch_tgt[0].to(device).float(), batch_tgt[1].to(device).long().reshape((-1,))

                optim_tgt.zero_grad()
                feature_src = self.target_encoder['sda'](images_src)
                pred_src = self.target_classifier['sda'](feature_src)
                feature_tgt = self.target_encoder['sda'](images_tgt)

                ce = ce_loss(pred_src, labels_src)
                csa = csa_loss(feature_src, feature_tgt, (labels_src == labels_tgt).float(), margin=self.config.margin)
                loss = (1 - self.config.alpha) * ce + self.config.alpha * csa

                loss.backward()
                optim_tgt.step()

                total_loss += loss.item()
                total_csa_loss += csa.item()
                total_ce_loss += ce.item()

            # Validation
            val_loss, val_acc = self._validate_adaptation(self.target_encoder['sda'], target_val_loader)
            
            # Add losses to the dictionary
            losses['total_loss'].append(total_loss / len(target_loader))
            losses['csa_loss'].append(total_csa_loss / len(target_loader))
            losses['ce_loss'].append(total_ce_loss / len(target_loader))
            losses['val_loss'].append(val_loss)
            losses['val_acc'].append(val_acc)

            self.logger.info(f'Epoch: {epoch} \tTotal Loss: {losses["total_loss"][-1]:.6f} \tCSA Loss: {losses["csa_loss"][-1]:.3f} \tTask loss: {losses["ce_loss"][-1]:.5f}')
            self.logger.info(f'Validation Loss: {val_loss:.6f} \tValidation Accuracy: {val_acc:.6f}')
            
            scheduler.step(val_loss)
            
            if epoch % self.config.save_interval == 0:
                self._save_models(method='sda', epoch=epoch)
            
            # if early_stopper.early_stop(val_loss):
            #     print(f"Early stopping after Epoch {epoch}")
            #     self._save_models(method='sda', epoch=epoch)
            #     break

        self._save_losses(losses, method='sda')

    def _validate_adaptation(self, encoder, val_loader):
        device = self.device
        ce_loss = nn.CrossEntropyLoss()
        val_loss = 0.
        val_acc = 0.

        encoder.eval()
        self.source_classifier.eval()

        with torch.no_grad():
            for batch in val_loader:
                data_val, labels_val = batch[0].to(device).float(), batch[1].to(device).long().reshape((-1,))
                preds_val = self.source_classifier(encoder(data_val))
                _, outputs_val = torch.max(preds_val.data, 1)
                correct = (outputs_val == labels_val).float().sum()
                loss = ce_loss(preds_val, labels_val)

                val_loss += loss.item()
                val_acc += correct.item() / data_val.shape[0]

        return val_loss / len(val_loader), val_acc / len(val_loader)

    def run_adaptation(self, source_loader, target_loader, target_val_loader):
        if "adda" in self.config.DA_methods:
            self._adapt_adda(source_loader, target_loader, target_val_loader)
        if "wdgrl" in self.config.DA_methods:
            self._adapt_wdgrl(source_loader, target_loader, target_val_loader)
        if "sda" in self.config.DA_methods:
            self._adapt_sda(source_loader, target_loader, target_val_loader)
        # Add other adaptation methods as needed

    def evaluate(self, test_loader, loaded_methods, desired_fpr=0.01):
        evaluation_results = {}
        device = self.device

        # Get the total number of samples in the test loader
        total_samples = len(test_loader.dataset)

        # print("shapes", self.source_classifier.input_dim, *[self.target_classifier[method].input_dim for method in config.DA_methods])

        for method in ['source'] + loaded_methods:
            self.logger.info(f"Evaluating {method} model")
            evaluation_results[method] = {
                'preds': np.empty(0, dtype=int), #total_samples
                'labels': np.empty(0, dtype=int),
                'filenames': np.empty(0, dtype=object),
                'scores': np.empty((0, 2), dtype=float),
                'encodings': np.empty((0, 256), dtype=float)
            }

            if method == 'source':
                encoder = self.source_encoder
                classifier = self.source_classifier
            else:
                encoder = self.target_encoder[method]
                classifier = self.target_classifier[method]

            encoder.eval()
            classifier.eval()

            start_idx = 0
            with torch.no_grad():
                for batch in test_loader:
                    data, labels, filenames = batch[0].float().to(device), batch[1].to(device), batch[2]
                    batch_size = data.size(0)

                    print("First batch")
                    print(batch[1].cpu().numpy())
                    
                    features = encoder(data)
                    outputs = classifier(features)
                    scores = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    end_idx = start_idx + batch_size
                    evaluation_results[method]['preds'] = np.append(evaluation_results[method]['preds'], preds.cpu().numpy())
                    evaluation_results[method]['labels'] = np.append(evaluation_results[method]['labels'], labels.cpu().numpy())
                    evaluation_results[method]['filenames'] = np.append(evaluation_results[method]['filenames'], np.array(filenames))
                    evaluation_results[method]['scores'] = np.append(evaluation_results[method]['scores'], scores.cpu().numpy(), axis=0)
                    evaluation_results[method]['encodings'] = np.append(evaluation_results[method]['encodings'], features.cpu().numpy(), axis=0)

                    invalid_labels_index = np.where((labels.cpu().numpy() != 0)|(labels.cpu().numpy() != 1))[0]
                    if invalid_labels_index.size > 0:
                        print(np.array(filenames)[invalid_labels_index])

                    # if method == "source":
                    #     for img, lb, fn, pr in zip(data, labels.cpu().numpy(), filenames, preds.cpu().numpy()):
                    #         plt.figure(figsize=(6,6))
                    #         plt.imshow(img.cpu().numpy()[1, :, :], origin='lower')
                    #         plt.colorbar()
                    #         plt.savefig(f'test_data_default/{lb}_{pr}_{fn[:-5]}.png', dpi=120)
                    #         plt.close()

                    start_idx = end_idx

            # Calculate metrics
            accuracy = accuracy_score(evaluation_results[method]['labels'], evaluation_results[method]['preds'])
            cm = confusion_matrix(evaluation_results[method]['labels'], evaluation_results[method]['preds'])
            print(np.unique(evaluation_results[method]['labels']), evaluation_results[method]['scores'][:, 1].max(), evaluation_results[method]['scores'][:, 1].min())
            
            evaluation_results[method]['fpr'], evaluation_results[method]['tpr'], evaluation_results[method]['roc_threshold'] = roc_curve(evaluation_results[method]['labels'], evaluation_results[method]['scores'][:, 1])
            evaluation_results[method]['roc_auc'] = auc(evaluation_results[method]['fpr'], evaluation_results[method]['tpr'])
            
            evaluation_results[method]['precision'], evaluation_results[method]['recall'], evaluation_results[method]['pr_threshold'] = precision_recall_curve(evaluation_results[method]['labels'], evaluation_results[method]['scores'][:, 1])

            evaluation_results[method]['f1_score'] = f1_score(evaluation_results[method]['labels'], evaluation_results[method]['preds'])


            # Calculate TPR at specific FPR values
            if desired_fpr is not None:
                if isinstance(desired_fpr, int) and desired_fpr > 1:
                    desired_fpr = desired_fpr / np.sum(evaluation_results[method]['labels'] == 0)
                elif isinstance(desired_fpr, float) and 0 < desired_fpr < 1:
                    pass  # Use the provided fraction directly
                else:
                    raise ValueError("desired_fpr must be an integer > 1 or a float between 0 and 1")

                threshold_idx = next((i for i, x in enumerate(evaluation_results[method]['fpr']) if x > desired_fpr), 0)
                threshold = evaluation_results[method]['roc_threshold'][threshold_idx]

                y_pred_threshold = (evaluation_results[method]['scores'][:, 1] > threshold).astype(int)
                cm_threshold = confusion_matrix(evaluation_results[method]['labels'], y_pred_threshold)

                self.logger.info(f"Confusion Matrix at FPR={desired_fpr:.4f}:\n{cm_threshold}")

            tpr_0 = next((t for f, t in zip(evaluation_results[method]['fpr'], evaluation_results[method]['tpr']) if f > 0), evaluation_results[method]['tpr'][0])
            tpr_10 = next((t for f, t in zip(evaluation_results[method]['fpr'], evaluation_results[method]['tpr']) if f > 10/np.sum(evaluation_results[method]['labels'] == 0)), evaluation_results[method]['tpr'][-1])

            evaluation_results[method]['tpr_0'] = tpr_0
            evaluation_results[method]['tpr_10'] = tpr_10

            # Print results
            self.logger.info(f"{method} Evaluation Results:")
            self.logger.info(f"Accuracy: {accuracy:.3f}")
            self.logger.info(f"Confusion Matrix:\n{cm}")
            self.logger.info(f"ROC AUC: {evaluation_results[method]['roc_auc']:.3f}")
            self.logger.info(f"TPR at FPR=0: {evaluation_results[method]['tpr_0']:.3f}")
            self.logger.info(f"TPR at FPR=10/N: {evaluation_results[method]['tpr_10']:.3f}")
            # print(f"Precision: {evaluation_results[method]['precision'][-1]:.3f}")
            # print(f"Recall: {evaluation_results[method]['recall'][-1]:.3f}")
            self.logger.info(f"F1 Score: {evaluation_results[method]['f1_score']:.3f}")
            self.logger.info("\n")

        self._save_evaluation_results(evaluation_results)

        return evaluation_results

    def _save_evaluation_results(self, evaluation_results):
        results_dir = os.path.join(self.config.output_dir, f'{self.config.encoder_name}', 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
        iteration_str = f'_{self.config.iteration}' if hasattr(self.config, 'iteration') else ''
        savename = f'_{self.config.savename}' if hasattr(self.config, 'savename') else ''
        results_file = os.path.join(results_dir, f'{self.config.encoder_name}_evaluation_results{iteration_str}{savename}.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(evaluation_results, f)
        self.logger.info(f"Evaluation results saved to {results_file}")

    def load_source_weights(self):
        source_encoder_path = os.path.join(self.config.output_dir, f'{self.config.encoder_name}', 'weights', 'source', 
                                         f'{self.config.encoder_name}_encoder_source_{self.config.iteration}.pth')
        source_classifier_path = os.path.join(self.config.output_dir, f'{self.config.encoder_name}', 'weights', 'source', 
                                            f'{self.config.encoder_name}_classifier_source_{self.config.iteration}.pth')
        
        if os.path.exists(source_encoder_path) and os.path.exists(source_classifier_path):
            try:
                self.source_encoder.load_state_dict(torch.load(source_encoder_path))
                self.source_classifier.load_state_dict(torch.load(source_classifier_path))
                self.logger.info(f"Loaded source weights for iteration {self.config.iteration}")
            except TypeError:
                self.source_classifier = torch.load(source_classifier_path)
                self.source_encoder.load_state_dict(torch.load(source_encoder_path), strict=False)
                self.logger.info(f"Loaded source weights for iteration {self.config.iteration}")
        else:
            raise FileNotFoundError(f"Source weights not found for iteration {self.config.iteration}")

    def load_target_weights(self):
        loaded_methods = []
        for method in self.config.DA_methods:
            # Define the base filenames without epoch
            base_encoder_filename = f'{self.config.encoder_name}_encoder_{method}'
            base_classifier_filename = f'{self.config.encoder_name}_classifier_{method}'
            
            # Define the directory where weights are stored
            weights_dir = os.path.join(self.config.output_dir, f'{self.config.encoder_name}', 'weights', method)
            self.logger.info("TEST LOADING WEIGHTS")
            self.logger.info(f"{base_encoder_filename}, {base_classifier_filename}, {weights_dir}")
            
            # Function to find the latest epoch file
            def find_latest_epoch_file(base_filename):
                pattern = re.compile(f"{base_filename}(_epoch_\d+)?_{self.config.iteration}.pth")
                matching_files = glob.glob(os.path.join(weights_dir, f"{base_filename}*_{self.config.iteration}.pth"))
                if not matching_files:
                    return None
                
                latest_file = max(matching_files, key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1) if re.search(r"epoch_(\d+)", f) else 0))
                return latest_file

            # Find the latest epoch files
            target_encoder_path = find_latest_epoch_file(base_encoder_filename)
            target_classifier_path = find_latest_epoch_file(base_classifier_filename)

            self.logger.info(f"{target_encoder_path}, {target_classifier_path}")
            
            if target_encoder_path and target_classifier_path:
                try:
                    self.target_encoder[method].load_state_dict(torch.load(target_encoder_path), strict=False)
                    self.target_classifier[method].load_state_dict(torch.load(target_classifier_path))
                    self.logger.info(f"Loaded {method} target weights: \nEncoder: {os.path.basename(target_encoder_path)}\nClassifier: {os.path.basename(target_classifier_path)}")
                    loaded_methods.append(method)
                except TypeError:
                    self.target_encoder[method].load_state_dict(torch.load(target_encoder_path), strict=False)
                    self.target_classifier[method] =  torch.load(target_classifier_path)
                    loaded_methods.append(method)
                    self.logger.info(f"Loaded {method} target weights: \nEncoder: {os.path.basename(target_encoder_path)}\nClassifier: {os.path.basename(target_classifier_path)}")
            else:
                warnings.warn(f"{method} target weights not found for iteration {self.config.iteration}. This method will not be available for evaluation.")
        
        if not loaded_methods:
            raise ValueError("No target weights could be loaded. Cannot proceed with evaluation.")
        
        return loaded_methods


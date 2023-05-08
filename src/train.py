from data_loader import ImageDataLoader
from transformations import get_train_transform, get_test_transform
from utils import set_gpu_mode, initialize_wandb, get_balanced_dataset_sampler
from models.model_factory import get_model

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torch

class Config:
    def __init__(self, model_name, k_folds, num_epochs, data_dir, wandb_on, lr) -> None:
        self.model_name = model_name
        self.k_folds = k_folds
        self.num_epochs = num_epochs
        self.data_dir = data_dir
        self.wandb_on = wandb_on
        self.lr = lr

config = Config(
    model_name='EfficientNet-b0',
    k_folds=5,
    num_epochs=2,
    data_dir='./data/binary/hiper_normal/',
    wandb_on=False,
    lr=1e-4
)



initialize_wandb(config)


data_loader = ImageDataLoader(config.data_dir)


skfold = StratifiedKFold(n_splits=config.k_folds, shuffle=True)
binary_labels = [sample[1] for sample in data_loader.dataset.samples]
bag = []
for fold, (train_ids, test_ids) in enumerate(skfold.split(data_loader.dataset, binary_labels)):
# K-fold Cross Validation model evaluation

    print(f'FOLD {fold +1}')
    print('--------------------------------')


    train_subset = Subset(data_loader.dataset, train_ids)
    train_subset.transform = get_train_transform()
    sampler = get_balanced_dataset_sampler(data_loader, train_ids, train_subset)
    train_loader = DataLoader(train_subset, batch_size=50, sampler=sampler)

    test_subset = Subset(data_loader.dataset, test_ids) 
    test_subset.transform = get_test_transform()
    test_loader = DataLoader(test_subset, batch_size=50, shuffle=True )
    

    model = get_model(config.model_name)
    set_gpu_mode(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_function = nn.BCELoss()

    for epoch in range(1, config.num_epochs+1):
        print(f'Starting epoch {epoch}')

        
        current_loss = 0.0
        for i, data in enumerate(train_loader):
            # Get inputs
            inputs, targets = data
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)


        # for img, labels in train_loader:
        #     num_retrievers += torch.sum(labels == 0)
        #     num_elkhounds += torch.sum(labels == 1)

    
    # print(num_retrievers.item())
    # print(num_elkhounds.item())
    if fold == 0:
        break




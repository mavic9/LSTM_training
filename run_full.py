import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from LSTM import LSTM_model, init_weights, count_parameters, get_optimizer, DynamicStepScheduler
from getData import getTable, collate_fn_factory, CustomDataset


#######
#Argument parser.
#######
parser=argparse.ArgumentParser(description="Training LSTM models for forecasting SynCom.")
parser.add_argument("-s","--path", help="Path to save directory.", default="Results")
parser.add_argument("-v","--name", help="Name of the model", default="model")
parser.add_argument("-t", "--data", help="Path to data table in tsv format.", default="")
parser.add_argument("-e", "--epochs", help="Number of training epochs.", default=2000)
parser.add_argument("-l", "--layers", help="Add additional layers.", default=1)
parser.add_argument("-b", "--batch", help="Batch size.", default=1024)
parser.add_argument("-d", "--back", help="The number of background days", default=128)
parser.add_argument("-n", "--nodes", help="The number of LSTM nodes in hidden layer", default=256)
parser.add_argument("-p", "--ahead", help="The number of days to predict ahead.", default=10)
parser.add_argument("-q", "--dataset", help="The round of training from wrapper script", default=1)
parser.add_argument("-r", "--lr", help="The learning rate of AdamW optimizer.", default=0.001)
parser.add_argument("-c", "--device", help="Name of device for training.", default="cuda:0")
parser.add_argument("-w", "--warm", help="number of epochs with high learning rate", default=3)
args=parser.parse_args()


def set_seed(seed):
    """
    Set seed for reproducibility.
    
    Args:
        seed (int): Seed value to use
    """
    # Python's random module
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch
    import torch
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Additional settings for complete determinism
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    torch.cuda.empty_cache()

    seed = 125
    set_seed(seed)

    #Main arguments
    dataset_num = int(args.dataset) - 1
    batch_size = args.batch # 8192 # 4096
    # start_day = args.start_day
    look_back = 128
    days = 0
    features = 28
    ahead = int(args.ahead)
    lr = float(args.lr)
    print(lr)
    if args.data:
        path_to_train_set = f"{args.data}/train_set_{dataset_num}.tsv"
        path_to_test_set = f"{args.data}/val_set_{dataset_num}.tsv"
        use_test_set = True
    else:
        path_to_train_set = f"/netscratch/dep_psl/grp_rgo/vm/clean_tables/df_contam_t0.3_l5_cleaned_ra.tsv"
        use_test_set = False
    opt_name = "adamw"
    weight_decay = 0.001
    l2_lambda = 0.0 # 0.0001 0.00005
    batch_size = int(args.batch)
    seq_length = int(args.back)
    feature_size = 28
    ahead = ahead
    nodes_n = int(args.nodes)
    add_layer = int(args.layers)
    device_name = args.device
    activation = 'none'
    model_name = args.name
    path = args.path
    epochs = int(args.epochs)
    # warm_step = int(args.warm)
    warm_step = 3
    initial_values = 0.0
    batchs_per_epochs = 16
    alpha, beta = 0.6, 0.4
    start_day = 16
    shuffle = True
    step_days = 1
    gradient_norm = 1.0
    drop = True # in default, drop = False
    

    # Create a directory to save the model checkpoints
    save_dir = f"{path}/{model_name}/Dataset_{dataset_num}"
    checkpoints = f"{save_dir}/checkpoints"
    os.makedirs(checkpoints, exist_ok=True)

    # if use_test_set:
    #     path_to_train_set = f"/netscratch/dep_psl/grp_rgo/vm/Tables_583d_seed1234/train_set_{dataset_num}.tsv"
    #     path_to_test_set = f"/netscratch/dep_psl/grp_rgo/vm/Tables_583d_seed1234/val_set_{dataset_num}.tsv"
    # else:
    #     path_to_train_set = f"/netscratch/dep_psl/grp_rgo/vm/clean_tables/df_contam_t0.3_l5_cleaned_ra.tsv"

    test_set = getTable(path_to_test_set)
    train_set = getTable(path_to_train_set)
    columns = list(test_set.columns)
    print(columns)
    test_set = test_set[columns]
    train_set = train_set[columns]
    print(f"Train set: {train_set.shape}")
    print(f"Test set: {test_set.shape}")

    collate_fn = collate_fn_factory(look_back, features, initial_values=initial_values)
    train_dataset = CustomDataset(train_set, start_day=16, look_back=look_back, days=days, features=features, ahead=ahead, step=step_days, shuffle=shuffle)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=0, collate_fn=collate_fn, drop_last=False)
    if use_test_set:
        val_dataset = CustomDataset(test_set, start_day=16, look_back=look_back, days=days, features=features, val=False, ahead=ahead, step=step_days,shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn, shuffle=False, drop_last=False)


    # Instantiate model
    model = LSTM_model(input_shape=(seq_length, feature_size),
                       opt="AdamW",
                       activation=activation,
                       add_layer=add_layer, 
                       epochs=200, 
                       batch_size=batch_size, 
                       weight_decay=0.005, 
                       lrate=0.001, 
                       nodes_n=nodes_n, 
                       split=0.0, 
                       unroll=False, 
                       ahead=ahead, 
                       lstm_dropout=0.4, 
                       device=device_name,
                       )
    print(model)
    model.apply(init_weights)
    print(f"The model has {count_parameters(model):,} trainable parameters")



    #Main parameters


    # Regularization strength

    test_set_log = test_set.shape if use_test_set else 'No test set'
    with open(f'{save_dir}/loss.txt', 'w') as loss_file:
        loss_file.write(f"Dataset #{dataset_num}\nSeed: {seed}\n"
                        f"Batch_size: {batch_size}\nBatchs per epoch: {batchs_per_epochs}\nContext: {look_back} days\nStart context: {start_day} d\n" 
                        f"Prediction ahead, d: {ahead}\nInitial values in time series: {initial_values}\n"
                        f"Days per step: {step_days}\n"
                        f"Train set: {train_set.shape}\nTest set: {test_set_log}\nStart training the model\n"
                        f"Path train: {path_to_train_set}\nPath test: {path_to_test_set}\nStart training the model\n"
                        f"\n{model}\n" + f"Regularization strength, l2_lambda: {l2_lambda}\n"
                        f"Optimizer: {opt_name}, LR: {lr}, WD: {weight_decay}\n"
                        f"Loss coefficients: Alpha {alpha}, Beta {beta}\n\n"
                        f"Shuffle: {shuffle}\n"
                        f"Strain_mae_loss[:23] + all_tech_loss[23:]\n\n"
                       )


    # device = "cuda"
    device = torch.device(device_name)

    # device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
    # model = CreateModel()
    # model= nn.DataParallel(model,device_ids = [1, 3])

    loss_list = []
    val_loss_list = []

    # Optimizer
    # optimizer = get_optimizer(opt_name="adamw", model_parameters=model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = get_optimizer(opt_name=opt_name, model_parameters=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = DynamicStepScheduler(optimizer, initial_step_size=warm_step, warm=True, gamma=0.1)
    # scheduler = create_scheduler(optimizer, step_size=3, gamma=0.5)

    model.to(device)
    loss_fn = model.loss_fn
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_metric = 0
        total_metric2 = 0
        gradient_norms = []
        # running_loss = 0
        for batch_idx, (encoder_input, target) in enumerate(tqdm(train_loader), 1):
            # print(encoder_input.shape)
            encoder_input = encoder_input.to(device).float()
            # decoder_input = decoder_input.to(device).float()
            target = target.to(device).float()
            # print(target[:, -ahead:, :].size())
            
            optimizer.zero_grad()
            target = target[:, -ahead:, :]
            output = model(encoder_input)
        
            loss = alpha * loss_fn.strain_mae_loss(target, output) + beta * loss_fn.all_tech_loss(target, output)
            metric = loss_fn.strain_mae_loss(target, output)
            metric2 = loss_fn.all_tech_loss(target, output)
            
            loss.backward()

            #Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)

            #Check the gradient
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm(2).item()
            gradient_norms.append(total_norm)
            
            optimizer.step()
            
            total_loss += loss.sum().item()
            total_metric += metric.item()
            total_metric2 += metric2.item()

            #control batches per epoch
            if batch_idx == batchs_per_epochs:
                break

        # current learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        # avg_loss = total_loss / len(train_loader)
        avg_loss = total_loss / batch_idx
        avg_metric = total_metric / batch_idx
        avg_metric2 = total_metric2 / batch_idx
        print(f"Epoch [{epoch+1}/{epochs}], Loss, MAE: {avg_loss:.4f}, Metric MAE: {avg_metric:.4f}, Tech: {avg_metric2:.4f}, LR: {current_lr:.6f}")
        loss_list.append(avg_loss)
        print(f"Gradient norms: {np.mean(gradient_norms)}")
        # Validation
        if use_test_set:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_metric = 0
                val_metric2 = 0
                for val_idx, (encoder_input, target) in enumerate(val_loader, 1):
                    encoder_input = encoder_input.to(device).float()
                    target = target.to(device).float()
                    
                    output = model(encoder_input)
                    loss = alpha * loss_fn.strain_mae_loss(target, output) + beta * loss_fn.all_tech_loss(target, output)
                    metric = loss_fn.strain_mae_loss(target, output)
                    metric2 = loss_fn.all_tech_loss(target, output)
                    
                    val_loss += loss.item()
                    val_metric += metric.item()
                    val_metric2 += metric2.item()
                avg_val_loss = val_loss / val_idx
                avg_val_metric = val_metric / val_idx
                avg_val_metric2 = val_metric2 / val_idx
                print(f"Validation Loss, MAE: {avg_val_loss:.4f}, Val Metric MAE: {avg_val_metric:.4f}, Tech: {avg_val_metric2:.4f}, LR: {current_lr:.6f}")

                val_loss_list.append(avg_val_loss)
            
        # Save model checkpoint for the current epoch
        checkpoint_path = f'{checkpoints}/model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        # print(f"Model saved at {checkpoint_path}")

        with open(f'{save_dir}/loss.txt', 'a+') as loss_file:
            i = epoch
            loss_file.write(f"Epoch [{epoch+1}/{epochs}] Loss, MAE: {avg_loss:.4f} Metric: {avg_metric:.4f}, Tech: {avg_metric2:.4f}, LR: {current_lr:.6f}\n")
            if use_test_set:
                loss_file.write(f"Epoch [{epoch+1}/{epochs}] Val_Loss, MAE: {avg_val_loss:.4f} Metric: {avg_val_metric:.4f}, Tech: {avg_val_metric2:.4f}\n\n")
        if epoch % 100 == 0 or epoch == epochs - 1:
            plot_loss(loss_list, val_loss_list, save_dir)
    torch.cuda.empty_cache()

def plot_loss(loss_list, val_loss_list, save_dir):
    min_val_loss_epoch = np.argmin(val_loss_list) + 1
    plt.figure(figsize=(6, 4))
    plt.plot(loss_list, label='Training Loss')
    if val_loss_list:
        plt.plot(val_loss_list, label='Validation Loss')
    plt.plot(min_val_loss_epoch-1, val_loss_list[min_val_loss_epoch-1], 'ro', 
             label=f'Min Val Loss: {val_loss_list[min_val_loss_epoch-1]:.4f} at Epoch {min_val_loss_epoch}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 0.1)
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(f'{save_dir}/loss_plot.pdf', format='pdf')
    plt.close()

if __name__ == '__main__':
    main()
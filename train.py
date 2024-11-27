import torch
# import timm
import os
import sys
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model.backbones import build_model
from engine import *
from utils.util import *
from config.setting import train_setting
from utils.dataset import NPY_datasets, Polyp_datasets

import warnings
warnings.filterwarnings("ignore")

pth_path = []
def main(config: train_setting, resume: bool=False, pth_path=None, datasets=None):

    # print('#========Creating logger========#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    if resume:
        ckpt_dir = os.path.join(pth_path, 'ckpts')
    else:
        ckpt_dir = os.path.join(config.work_dir, 'ckpts')
    resume_model = os.path.join(ckpt_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    
    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(f'{config.work_dir}summary')

    log_config_info(config, logger)
    

    print('#========GPU init========#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    # torch.backends.cudnn.enabled = False
    set_seed(config.seed)
    torch.cuda.empty_cache()


    print(f'#========Preparing dataset: \033[91m{config.datasets}\033[0m========#')
    if config.datasets == 'polyp':
        train_dataset = Polyp_datasets(config.data_path, config, mode='train')
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            pin_memory=True, num_workers=config.num_workers
        )

        val_loader_dict = {}
        for dataset in config.polyp_datasets:
            val_dataset = Polyp_datasets(config.data_path, config, mode='val', test_datasets=dataset)
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False, pin_memory=True, 
                num_workers=config.num_workers, drop_last=True
            )
            val_loader_dict[dataset] = val_loader
    
    else:
        train_dataset = NPY_datasets(config.data_path, config, mode='train')
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            pin_memory=True, num_workers=config.num_workers
        )

        val_dataset = NPY_datasets(config.data_path, config, mode='val')
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, pin_memory=True,
            num_workers=config.num_workers, drop_last=True
        )



    print('#========Preparing Model========#')
    model_config = config.model_config
    params = {
        "num_classes": model_config['num_classes'],
        "input_channels": model_config['input_channels'],
        "depths": model_config['depths'],
        "depths_decoder": model_config['depths_decoder'],
        "drop_path_rate": model_config['drop_path_rate'],
        "encoder_ckpt_path": model_config['encoder_ckpt_path'],
        "decoder_ckpt_path": model_config['decoder_ckpt_path'],
    }

    model = build_model(model_name='vssm_mkla', **params)
    model.load_from()
    logger.info('#----------Model info----------#')
    logger.info(model)
    
    model.cuda()

    cal_params_flops(model, 256, logger)



    print('#========Preparing loss, opt, sch and amp========#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    if os.path.exists(resume_model):
        # print('#=======Resume Model and Other params=======#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    
    step = 0
    print('#=======Training=======#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        
        if config.datasets == 'polyp':
            loss_all = []
            for name in config.polyp_datasets:
                val_loader_t = val_loader_dict[name]
                loss_t = val_one_epoch(
                    val_loader_t,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config,
                    val_data_name=name
                )
                loss_all.append(loss_t)
            loss = np.mean(loss_all)
        else:
            loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch
        
        torch.save(
            {
               'epoch': epoch,
               'min_loss': min_loss,
               'min_epoch': min_epoch,
               'loss': loss,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict() 
            }, os.path.join(ckpt_dir, 'latest.pth')
        )
    if os.path.exists(os.path.join(ckpt_dir, 'best.pth')):
        print('#----------Testing----------#')
        if not resume:
            best_weight = torch.load(config.work_dir + 'ckpts/best.pth', map_location=torch.device('cpu'))
        else:
            best_weight = torch.load(pth_path + '/ckpts/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)

        if config.datasets == 'polyp':
            for name in config.polyp_datasets:
                val_loader_t = val_loader_dict[name]
                loss = test_one_epoch(
                    val_loader_t,
                    model, 
                    criterion,
                    logger,
                    config,
                    test_data_name=name
                )
        else:
            loss = test_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    with_cam=False
                )
        os.rename(
            os.path.join(ckpt_dir, 'best.pth'),
            os.path.join(ckpt_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )   


if __name__ == '__main__':
    config = train_setting
    main(config, resume=False, pth_path=pth_path)

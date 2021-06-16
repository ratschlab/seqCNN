import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import seq_dataset 


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
    
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
        return self


def validate(net, loader, device, criteria, embed_dist):
    labels = []
    preds = []
    with torch.no_grad():
        net.eval()
        running_loss = 0
        for (data1, data2, target) in loader:
            target = target.float() 
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            embed1, embed2 = net(data1), net(data2)
            dist = embed_dist(embed1, embed2)
            loss = criteria(dist, target)
            running_loss += loss.item()
            preds.append(dist.cpu().data.numpy())
            labels.append(target.cpu().data.numpy())
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        return running_loss/len(loader), labels, preds

    
def train_epoch(net, optim, train_loader, device, criteria, embed_dist):
    net.train()
    progress_bar = tqdm(train_loader, total=len(train_loader))
    running_loss = 0
    for it,(data1,data2,target) in enumerate(progress_bar):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        optim.zero_grad()
        embed1, embed2 = net(data1), net(data2)
        dist = embed_dist(embed1, embed2)
        loss = criteria(dist, target)
        running_loss += loss.item()
        if it % 10==0:
            progress_bar.set_description(f"running loss = {loss.item():.4f}")
        loss.backward()
        optim.step()

        
def train(net, optim, train_loader, val_loader, epochs, device, 
          earlyStopping, trWriter, valWriter, criteria, embed_dist, visualize):  
    img = visualize(net, val_loader, device)        
    valWriter.add_image("exact vs. embedding distance", img, 0) # epoch==0  => initial condition
    for epoch in range(1, epochs+1):
        train_epoch(net, optim, train_loader, device, criteria, embed_dist)    
        train_loss, _, _ = validate(net, train_loader, device, criteria, embed_dist)
        val_loss, labels, preds = validate(net, val_loader, device, criteria, embed_dist)
        
        trWriter.add_scalar("training vs. validation loss", train_loss, epoch)
        valWriter.add_scalar("training vs. validation loss", val_loss, epoch)
        img = visualize(net, val_loader, device)        
        valWriter.add_image("exact vs. embedding distance", img, epoch)
        for th in [.1, .2, .3]:
            valWriter.add_pr_curve(f"exact vs. embedding, ED thresh = {th}", labels>th, preds, epoch)
        print(f"epoch {epoch} train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
        
        if earlyStopping(val_loss).early_stop:
            break



def train_model(net_model, epochs, device, batchsize, lr, patience, min_delta, criteria, embed_dist,
              dataset_params, model_params, trWriter, valWriter, visualize):
    assert(dataset_params['alph']==model_params['in_channels'])
    net = net_model(**model_params).to(device)

    train_dataset, val_dataset = seq_dataset.train_val_datasets(**dataset_params)
    train_loader = DataLoader(dataset = train_dataset,batch_size = batchsize,shuffle=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batchsize,shuffle=False)

    optim = torch.optim.Adam(net.parameters(), lr)
    earlyStopping = EarlyStopping(patience=patience,min_delta=min_delta)
    

    train(
        net=net, 
        trWriter=trWriter,
        valWriter=valWriter,
        optim=optim, 
        criteria=criteria,
        embed_dist=embed_dist,
        earlyStopping=earlyStopping,
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=epochs, 
        device=device,
        visualize=visualize,
    )
    sample_data,_,_ = next(iter(val_loader))
    net = net.cpu()
    trWriter.add_graph(net, sample_data)
    hparams = {f'hp/{k}':v for k,v in model_params.items()}
    hparams.update({'hp/lr': lr, 
                    'hp/batchsize': batchsize})
    trWriter.add_hparams(hparams,
                         {'hp/val_loss': earlyStopping.best_loss})
    valWriter.close()
    trWriter.close() 
    
    
    return net, train_loader, val_loader

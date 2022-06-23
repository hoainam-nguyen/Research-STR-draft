from src.optim.optim import ScheduledOptim
from src.optim.loss import LossFunction
from torch.optim import Adam, SGD, AdamW
from torch import nn
from src.tools.utils import build_model
from src.tools.utils import translate, batch_to_device
from src.tools.logger import Logger
from src.loader.aug import ImgAugTransform

import yaml
import torch
from src.loader.dataloader import SceneTextDataset, ClusterRandomSampler, Collator
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, OneCycleLR

import torchvision 

from src.tools.utils import compute_accuracy
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class Trainer():
    def __init__(self, config, pretrained=False, augmentor=ImgAugTransform()):

        self.config = config
        self.model, self.vocab = build_model(config)
        
        self.device = config['device']
        self.num_iters = config['trainer']['iters']

        self.data_root = config['dataset']['data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.dataset_name = config['dataset']['name']

        self.train_lmdb = config['dataset']['train_lmdb']
        self.valid_lmdb = config['dataset']['valid_lmdb']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']
        
        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']
        
        self.export_weights = config['trainer']['export']
        self.metrics = config['trainer']['metrics']
        self.logger = Logger(config['trainer']['log']) 

        self.iter = 0
        
        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])
#        self.optimizer = ScheduledOptim(
#            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
#            #config['transformer']['d_model'], 
#            512,
#            **config['optimizer'])

        #self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
        self.criterion = LossFunction(self.config, len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
        transforms = None
        if self.image_aug:
            transforms =  augmentor

        self.train_gen = self.data_gen(lmdb_path = self.train_lmdb, 
                                       data_root =  self.data_root, 
                                       annotation = self.train_annotation, 
                                       masked_language_model = self.masked_language_model, 
                                       transform=transforms)
        if self.valid_annotation:
            self.valid_gen = self.data_gen(lmdb_path = self.valid_lmdb, 
                                           data_root = self.data_root, 
                                           annotation = self.valid_annotation, 
                                            masked_language_model=False)

        self.train_losses = []
        
    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = SceneTextDataset(lmdb_path=lmdb_path, 
                root_dir=data_root, annotation_path=annotation, 
                vocab=self.vocab, transform=transform, 
                image_height=self.config['dataset']['image_height'], 
                image_min_width=self.config['dataset']['image_min_width'], 
                image_max_width=self.config['dataset']['image_max_width'])

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
                dataset,
                batch_size=self.batch_size, 
                sampler=sampler,
                collate_fn = collate_fn,
                shuffle=False,
                drop_last=False,
                **self.config['dataloader'])
       
        return gen

    def step_train(self, batch):
        self.model.train()
        batch = batch_to_device(self.device, batch)

        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']    
        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)

        #outputs = outputs.view(-1, outputs.size(2)) #flatten(0, 1)
        tgt_output = tgt_output.view(-1) #flatten()
        
        loss = self.criterion(outputs, tgt_output)
        #print('outpus', outputs[0])
        self.optimizer.zero_grad()

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) 

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item

    def train(self):
        total_loss = 0
        
        total_loader_time = 0
        total_gpu_time = 0
        best_acc = 0

        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step_train(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter, 
                        total_loss/self.print_every, self.optimizer.param_groups[0]['lr'], 
                        total_loader_time, total_gpu_time)
                self.logger.log(info)
                if self.iter % self.print_every == 0:
                    total_loss = 0
                    total_loader_time = 0
                    total_gpu_time = 0
                    print(info) 
                    self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char = self.precision(self.metrics)

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(self.iter, val_loss, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)

                if acc_full_seq > best_acc:
                    self.save_weights(self.export_weights)
                    best_acc = acc_full_seq
        
        print('Done training')
        self.save_checkpoint('./weights/final_weght.pth')
            
    def validate(self):
        self.model.eval()

        total_loss = []
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                #batch = self.batch_to_device(batch)
                batch = batch_to_device(self.device, batch)
                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

                outputs = self.model(img, tgt_input, tgt_padding_mask)
  
                #outputs = outputs.flatten(0,1)
                tgt_output = tgt_output.flatten()
                loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())
                
                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        
        return total_loss
    
    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []

        for batch in  self.valid_gen:
            batch = batch_to_device(self.device, batch)
            #batch = self.batch_to_device(batch)

            translated_sentence, prob = translate(batch['img'], self.model)
            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())

            img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            
            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, prob

    def precision(self, sample=None):

        pred_sents, actual_sents, _, _ = self.predict(sample=sample)

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
    
        return acc_full_seq, acc_per_char

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        optim = ScheduledOptim(
	       Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            	self.config['transformer']['d_model'], **self.config['optimizer'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.iter = checkpoint['iter']

        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter':self.iter, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        
        # path, _ = os.path.split(filename)
        # os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
       
        torch.save(self.model.state_dict(), filename)
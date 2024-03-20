import torch
import numpy as np
import copy
from tqdm import tqdm

def train_earlystop(device,model,loss_fn,optimizer,train_loader,val_loader,test_loader,num_epochs=100):
    # Training Loop
    print('\nStarting to Train {} for {} Epochs!'.format(model.name,num_epochs))
    train_loss = []
    val_loss = []
    val_acc = 0
    best_model = model
    min_val_loss = 0

    for epoch_idx in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_count = 0
        train_correct_count = 0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)
            optimizer.zero_grad()
            logits = model(train_x)
            loss = loss_fn(logits, train_y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                y_hat = torch.argmax(logits, dim=-1)
                y = torch.argmax(train_y,dim=-1)
                train_correct_count += torch.sum(y_hat == y, axis=-1)
                train_count += train_x.size(0)
        train_loss.append(loss.cpu().detach().numpy())
        train_acc = train_correct_count / train_count
    
        # Validation
        model.eval()
        val_count = 0
        val_correct_count = 0
        with torch.no_grad():
            for idx, (val_x, val_y) in enumerate(val_loader):
                val_x = val_x.float().to(device)
                val_y = val_y.float().to(device)
                logits = model(val_x).detach()
                loss = loss_fn(logits, val_y)
                y_hat = torch.argmax(logits, dim=-1)
                y = torch.argmax(val_y,dim=-1)
                val_correct_count += torch.sum(y_hat == y, axis=-1)
                val_count += val_x.size(0)
        val_loss.append(loss.cpu().detach().numpy())
        val_acc = val_correct_count / val_count
        # Store the Model with the Lowest Val Loss into the Best Model
        if(val_loss[epoch_idx] <= min_val_loss):
          max_val_acc = val_loss[epoch_idx]
          best_model = copy.deepcopy(model)
        print('Epoch [{}/{}]: Train Loss: {:.3f} Val Loss: {:.3f} Train Acc: {:.3f}, Val Acc: {:.3f}'.format(epoch_idx+1,num_epochs,train_loss[epoch_idx], val_loss[epoch_idx],train_acc, val_acc))

    return train_loss, val_loss, min_val_loss, best_model

def train(device,model,loss_fn,optimizer,train_loader,val_loader,test_loader,num_epochs=100):
    # Training Loop
    print('\nStarting to Train {} for {} Epochs!'.format(model.name,num_epochs))
    train_loss = []
    val_loss = []
    for epoch_idx in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_count = 0
        train_correct_count = 0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)
            optimizer.zero_grad()
            logits = model(train_x)
            loss = loss_fn(logits, train_y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                y_hat = torch.argmax(logits, dim=-1)
                y = torch.argmax(train_y,dim=-1)
                train_correct_count += torch.sum(y_hat == y, axis=-1)
                train_count += train_x.size(0)
        train_loss.append(loss.cpu().detach().numpy())
        train_acc = train_correct_count / train_count
    
        # Validation
        model.eval()
        val_count = 0
        val_correct_count = 0
        with torch.no_grad():
            for idx, (val_x, val_y) in enumerate(val_loader):
                val_x = val_x.float().to(device)
                val_y = val_y.float().to(device)
                logits = model(val_x).detach()
                loss = loss_fn(logits, val_y)
                y_hat = torch.argmax(logits, dim=-1)
                y = torch.argmax(val_y,dim=-1)
                val_correct_count += torch.sum(y_hat == y, axis=-1)
                val_count += val_x.size(0)
        val_loss.append(loss.cpu().detach().numpy())
        val_acc = val_correct_count / val_count
    
        print('Epoch [{}/{}]: Train Loss: {:.3f} Val Loss: {:.3f} Train Acc: {:.3f}, Val Acc: {:.3f}'.format(epoch_idx+1,num_epochs,train_loss[epoch_idx], val_loss[epoch_idx],train_acc, val_acc))
    
    return train_loss, val_loss

def eval(device,model,test_loader):
    model.eval()
    with torch.no_grad():
        test_count = 0
        test_correct_count = 0
        for idx, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.float().to(device)
            test_y = test_y.float().to(device)
            logits = model(test_x).detach()
            y_hat = torch.argmax(logits, dim=-1)
            y = torch.argmax(test_y,dim=-1)
            test_correct_count += torch.sum(y_hat == y, axis=-1)
            test_count += test_x.size(0)
        test_acc = test_correct_count / test_count
        return test_acc.cpu().detach().numpy()  
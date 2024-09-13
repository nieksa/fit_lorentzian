import torch
import copy
from draw import draw_prediction

def test_model(model,device, test_loader, criterion):
    model.eval()  # 切换到评估模式
    test_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
    
    test_loss /= len(test_loader.dataset)
    return test_loss


def start_training(model,device,num_epochs,criterion,optimizer,train_loader,val_loader,test_loader,early_stopping):
    best_val_loss = float('inf')
    best_model_state = None
    print(f"training with {device}")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            train_loss += loss.item() * batch_x.size(0)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss : {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch:{epoch}")
            break
        
        # 保存当前损失最小的模型权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_model_epoch = epoch

    model.load_state_dict(torch.load('checkpoint.pt'))

    torch.save(best_model_state, f'best_model_epoch_{best_model_epoch}.pth')
    torch.save(model.state_dict(), 'last_model_weights.pth')

    # 测试最优模型
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(f'best_model_epoch_{best_model_epoch}.pth'))
    test_loss_best = test_model(best_model, device, test_loader, criterion)

    # 测试最后一个模型
    last_model = copy.deepcopy(model)
    last_model.load_state_dict(torch.load('last_model_weights.pth'))
    test_loss_last = test_model(last_model, device, test_loader, criterion)


    print(f'Test Loss of Best Model: {test_loss_best:.4f}')
    print(f'Test Loss of Last Model: {test_loss_last:.4f}')

    # if early_stopping.early_stop:
    #     model.load_state_dict(torch.load('checkpoint.pt'))
    #     test_loss_stopped = test_model(model, device, test_loader, criterion)
    #     print(f'Test Loss of Early_stopped Model: {test_loss_stopped:.4f}')

    draw_prediction(best_model)
    draw_prediction(last_model)
    # if early_stopping.early_stop:
    #     draw_prediction(model)
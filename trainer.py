import torch
from torch import save, load
from torch.nn import MSELoss, BCELoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
# from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloaders
from model import Shift_model
import os


def loss_function(out, label):
    l1 = BCELoss()(out[:, 0], label[:, 0])
    l2 = MSELoss()(out[:, 1:], label[:, 1:])
    loss_tot = l1 * 1 +\
               l2 * 10
    return loss_tot

if __name__=='__main__' :

    data_path = '/home/nicolas/unsupervised-detection/dataset/general/'
    models_path = './models/'

    epochs_already_trained = 0
    batch_size = 128
    epochs_nb = 200
    lr = 1e-3
    size = (256, 172)

    model = Shift_model().cuda()

    train_dataloader, test_dataloader = get_dataloaders(data_path, size, batch_size=batch_size, train_test_ratio=0.8)
    test_dataloader.dataset.dataset.set_augment(False)

    optimizer = Adam(model.parameters(),
                     lr=lr,
                     weight_decay=1e-5)
    scheduler1 = MultiStepLR(optimizer, milestones=[10, 50, 150], gamma=0.1)
    scheduler2 = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True, threshold=1e-3, min_lr=1e-5)

    if not os.path.isdir(models_path) : os.mkdir(models_path)

    if epochs_already_trained != 0:
        model.load_state_dict(load(models_path+'shift.pth'))

    best_loss = 1000

    for epoch in range(epochs_already_trained, epochs_already_trained + epochs_nb) :
        total_epoch_loss = 0
        for batch in train_dataloader :
            img = batch['img'].cuda()
            label = batch['label'].cuda()

            out = model.forward(img)
            loss = loss_function(out, label)
            total_epoch_loss += float(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_epoch_loss /= len(train_dataloader)
        # writer.add_scalar('Loss', total_epoch_loss, epoch + 1)
        print('train :', total_epoch_loss, epoch+1)

        with torch.no_grad() :
            total_epoch_loss = 0
            for batch in test_dataloader :
                img = batch['img'].cuda()
                label = batch['label'].cuda()

                out = model.forward(img)
                loss = loss_function(out, label)
                total_epoch_loss += float(loss)

            total_epoch_loss /= len(test_dataloader)
            # writer.add_scalar('Loss', total_epoch_loss, epoch + 1)
            print('test :', total_epoch_loss, epoch + 1)
            print()

        scheduler1.step()
        scheduler2.step(total_epoch_loss)

        if best_loss > total_epoch_loss :
            best_loss = total_epoch_loss
            save(model.state_dict(), models_path+'/shift.pth')
            print('Saved at epoch ' + str(epoch+1),'\n')
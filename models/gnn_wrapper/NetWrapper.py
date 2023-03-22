import time
from datetime import timedelta
import torch
from torch import optim
from sklearn.metrics import roc_auc_score

def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"


class NetWrapper:

    def __init__(self, model, loss_function, device='cpu', classification=True, config={}):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification
        self.config = config
        self.roc_auc = self.config['roc_auc'] if 'roc_auc' in self.config else False

    def _train(self, train_loader, optimizer, clipping=None):
        model = self.model.to(self.device)

        model.train()

        loss_all = 0
        acc_all = 0
        auc_roc = 0
        for i, data in enumerate(train_loader):
            if i == len(train_loader)-1:
                continue
            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            if self.classification:
                loss, acc = self.loss_fun(data.y, *output)
                loss.backward()
                try:
                    num_graphs = data.num_graphs
                except TypeError:
                    num_graphs = data.adj.size(0)

                loss_all += loss.item() * num_graphs
                acc_all += acc.item() * num_graphs
                if self.roc_auc:
                    auc_roc += roc_auc_score(data.y.detach().cpu().numpy(), torch.argmax(output[0], dim=-1).detach().cpu().numpy())
            else:
                loss = self.loss_fun(data.y, *output)
                loss.backward()
                loss_all += loss.item()
            
            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()

        # TODO: add auc-roc

        if self.classification:
            if self.roc_auc:
                return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset), auc_roc/len(train_loader.dataset)
            else:
                return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)
        else:
            return None, loss_all / len(train_loader.dataset)

    def classify_graphs(self, loader):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        acc_all = 0
        auc_roc = 0

        for data in loader:
            data = data.to(self.device)
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            if self.classification:
                loss, acc = self.loss_fun(data.y, *output)

                try:
                    num_graphs = data.num_graphs
                except TypeError:
                    num_graphs = data.adj.size(0)

                loss_all += loss.item() * num_graphs
                acc_all += acc.item() * num_graphs
                if self.roc_auc:
                    auc_roc += roc_auc_score(data.y.detach().cpu().numpy(), torch.argmax(output[0], dim=-1).detach().cpu().numpy())
            else:
                loss = self.loss_fun(data.y, *output)
                loss_all += loss.item()

        if self.classification:
            if self.roc_auc:
                return acc_all / len(loader.dataset), loss_all / len(loader.dataset), auc_roc / len(loader.dataset)
            else:
                return acc_all / len(loader.dataset), loss_all / len(loader.dataset)
        else:
            return None, loss_all / len(loader.dataset)

    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=1):

        early_stopper = early_stopping() if early_stopping is not None else None

        val_loss, val_acc = -1, -1
        test_loss, test_acc = None, None
        train_roc_auc,val_roc_auc,test_roc_auc = -1, -1, -1

        time_per_epoch = []

        for epoch in range(1, max_epochs+1):

            start = time.time()
            if self.roc_auc:
                train_acc, train_loss, train_roc_auc = self._train(train_loader, optimizer, clipping)
            else:
                train_acc, train_loss = self._train(train_loader, optimizer, clipping)

            # TODO: calculate norm before clipping 
            total_norm = 0.0
            print('model summary:', self.model)
            for p in self.model.parameters():
                if p.grad is None:
                    param_norm = p.norm(2)
                else:
                    param_norm = p.grad.data.norm(2)
                    
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            end = time.time() - start
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step(epoch)

            if test_loader is not None:
                if self.roc_auc:
                    test_acc, test_loss, test_roc_auc = self.classify_graphs(test_loader)
                else:
                    test_acc, test_loss = self.classify_graphs(test_loader)

            if validation_loader is not None:
                if self.roc_auc:
                    val_acc, val_loss, val_roc_auc = self.classify_graphs(validation_loader)
                else:
                    val_acc, val_loss = self.classify_graphs(validation_loader)

                # Early stopping (lazy if evaluation)
                if early_stopper is not None and early_stopper.stop(epoch, val_loss, val_acc,
                                                                    test_loss, test_acc,
                                                                    train_loss, train_acc):
                    msg = f'Stopping at epoch {epoch}, best is {early_stopper.get_best_vl_metrics()}'
                    if logger is not None:
                        logger.log(msg)
                        print(msg)
                    else:
                        print(msg)
                    break

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR acc: {train_acc}, VL loss: {val_loss} VL acc: {val_acc} ' \
                    f'TE loss: {test_loss} TE acc: {test_acc}, GradNorm: {total_norm}, TR rocauc: {train_roc_auc}, VL rocauc:{val_roc_auc}, \
                        TE rocauc: {test_roc_auc}'
                    # TODO: add grad norm
                    
                
                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())

        elapsed = format_time(avg_time_per_epoch)

        if early_stopper is not None:
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, best_epoch = early_stopper.get_best_vl_metrics()

        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, elapsed

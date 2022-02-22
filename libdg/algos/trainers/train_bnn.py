from libdg.algos.trainers.a_trainer import TrainerClassif
import torch.optim as optim


class TrainerBnn(TrainerClassif):
    def __init__(self, model, task, observer, device, aconf=None):
        super().__init__(model, task, observer, device, aconf)
        self.optimizer = optim.Adam(self.model.parameters(), lr=aconf.lr)
        self.epo_loss_tr = None
        self.num_repeats = 100

    def tr_epoch(self, epoch):
        self.model.train()
        self.epo_loss_tr = 0
        batches = len(self.loader_tr)
        for i, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x = tensor_x.repeat(self.num_repeats, 1, 1, 1)  # FIXME
            vec_y = vec_y.repeat(self.num_repeats, 1)
            vec_d = vec_d.repeat(self.num_repeats, 1)
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            recon, kl = self.model(tensor_x, vec_y, vec_d)
            coeff = (2**(batches - i)) / (2**(batches) - 1)
            loss = recon + coeff * kl
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
        print("epoch loss aggregated:", self.epo_loss_tr)
        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop

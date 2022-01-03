from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.utils.perf_cluster import PerfCluster

class ObVisitorClustering(ObVisitor):
    """
    Observer + Visitor pattern for clustering algorithms
    """

    def update(self, epoch):
        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            acc_tr_pool = PerfCluster.cal_acc(self.host_trainer.model, self.loader_tr, self.device)
            print("pooled train clustering acc: ", acc_tr_pool)
            acc_te = PerfCluster.cal_acc(self.host_trainer.model, self.loader_te, self.device)
            self.acc_te = acc_te
            print("out of clustering test acc: ", acc_te)
        return super.update()

    def after_all(self):
        """
        After training is done
        """
        super.after_all()
        model_ld = self.exp.visitor.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()
        acc_te = PerfCluster.cal_acc(model_ld, self.loader_te, self.device)
        print("persisted model clustering acc: ", acc_te)

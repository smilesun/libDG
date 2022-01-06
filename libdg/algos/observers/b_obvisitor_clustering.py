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
            # FIXME: loader_te does not have the same clusters as loader_tr (nor does it have the same number of clusters), so the following cannot be computed.. We need to adjust what loader_te is for clustering experiments...
            #acc_te = PerfCluster.cal_acc(self.host_trainer.model, self.loader_te, self.device)
            acc_te = None
            self.acc_te = acc_te
            print("clustering test acc: ", acc_te)
        return super().update(epoch)

    def after_all(self):
        """
        After training is done
        """
        super().after_all()
        model_ld = self.exp.visitor.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()
        # FIXME: (Same comment as above) loader_te does not have the same clusters as loader_tr (nor does it have the same number of clusters), so the following cannot be computed.. We need to adjust what loader_te is for clustering experiments...
        #acc_te = PerfCluster.cal_acc(model_ld, self.loader_te, self.device)
        acc_te = None
        print("persisted model clustering acc: ", acc_te)

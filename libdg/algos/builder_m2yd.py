from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.trainers.train_basic import TrainerBasic
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from libdg.models.model_m2yd import ModelXY2D
from libdg.utils.utils_cuda import get_device


class NodeAlgoBuilderM2YD(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)
        model = ModelXY2D(y_dim=len(task.list_str_y),
                          list_str_y=task.list_str_y,
                          zd_dim=args.zd_dim,
                          gamma_y=args.gamma_y,
                          device=device,
                          i_c=task.isize.c,
                          i_h=task.isize.h,
                          i_w=task.isize.w)
        observer = ObVisitorCleanUp(
            ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        trainer = TrainerBasic(model, task, observer, device, aconf=args)
        return trainer

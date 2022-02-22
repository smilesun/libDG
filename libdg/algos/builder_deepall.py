from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.trainers.train_basic import TrainerBasic
from libdg.algos.trainers.train_bnn import TrainerBnn
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.observers.bnn_obvisitor import BnnVisitor
from libdg.utils.utils_cuda import get_device
from libdg.compos.nn_alex import Alex4DeepAll
from libdg.models.model_deep_all import ModelDeepAll, ModelBNN
from libdg.utils.u_import import import_path


class NodeAlgoBuilderDeepAll(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)
        if args.mpath is not None:
            net_module = import_path(args.mpath)
            if hasattr(net_module, "build_net"):
                try:
                    net = net_module.build_net(task.dim_y, task.isize.i_c, task.isize.i_h, task.isize.i_w)
                except Exception:
                    print("function build_net(dim_y, i_c, i_h, i_w) should return a neural network that \
                          that classifies dim_y classes, with image channel i_c, height i_h, width i_w")
                    raise

                if net is None:
                    raise RuntimeError("Please return the pytorch module you have implemented in build_net")
            else:
                raise RuntimeError("Please implement a function build_net \
                                   in your python file refered by -mpath")
            if hasattr(net, "probforward"):
                model = ModelBNN(net, list_str_y=task.list_str_y,
                                 net_builder=net_module.build_net, task=task)
            else:
                model = ModelDeepAll(net, list_str_y=task.list_str_y)

        else:
            net = Alex4DeepAll(flag_pretrain=True, dim_y=task.dim_y)
            model = ModelDeepAll(net, list_str_y=task.list_str_y)

        observer = ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)
        if hasattr(net, "probforward"):
            observer = BnnVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)
            trainer = TrainerBnn(model, task, observer, device, args)
        else:
            trainer = TrainerBasic(model, task, observer, device, args)
        return trainer

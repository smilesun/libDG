from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.trainers.train_basic import TrainerBasic
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.utils.utils_cuda import get_device
from libdg.compos.nn_alex import AlexNetNoLastLayer
from libdg.compos.net_classif import ClassifDropoutReluLinear
from libdg.models.model_dann import ModelDAN


class NodeAlgoBuilderDANN(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)
        observer = ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)
        net_encoder = AlexNetNoLastLayer(flag_pretrain=True)
        model = ModelDAN(list_str_y=task.list_str_y, list_str_d=task.list_domain_tr,
                         alpha=1.0,
                         net_encoder=net_encoder,
                         net_classifier=ClassifDropoutReluLinear(4096, task.dim_y),
                         net_discriminator=ClassifDropoutReluLinear(4096, len(task.list_domain_tr)))
        trainer = TrainerBasic(model, task, observer, device, args)
        return trainer

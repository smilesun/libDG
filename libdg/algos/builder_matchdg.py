from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.trainers.train_matchdg import TrainerMatchDG
from libdg.compos.nn_alex import AlexNetNoLastLayer, Alex4DeepAll
from libdg.models.model_deep_all import ModelDeepAll
from libdg.models.wrapper_matchdg import ModelWrapMatchDGLogit
from libdg.utils.utils_cuda import get_device


class NodeAlgoBuilderMatchDG(NodeAlgoBuilder):
    """
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)
        fun_build_ctr, fun_build_erm_phi = \
            get_ctr_model_erm_creator(task, args)
        model = fun_build_erm_phi()
        model = model.to(device)
        ctr_model = fun_build_ctr()
        ctr_model = ctr_model.to(device)
        observer = ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)
        trainer = TrainerMatchDG(exp, task, ctr_model, model, observer, args, device)
        return trainer


def get_ctr_model_erm_creator(task, args):
    def fun_build_alex_erm():
        #net = Alex4DeepAll(flag_pretrain=True, dim_y=dim_y)
        #model = ModelDeepAll(net, list_str_y=list_str_y)
        from libdg.compos.pcr.request import RequestVAEBuilderCHW
        from libdg.models.model_hduva import ModelHDUVA
        from libdg.utils.utils_cuda import get_device
        from libdg.compos.vae.utils_request_chain_builder import VAEChainNodeGetter


        request = RequestVAEBuilderCHW(task.isize.c, task.isize.h, task.isize.w)
        device = get_device(args.nocu)
        node = VAEChainNodeGetter(request, args.topic_dim)()
        model = ModelHDUVA(node,
                           zd_dim=args.zd_dim,
                           zy_dim=args.zy_dim,
                           zx_dim=args.zx_dim,
                           device=device,
                           topic_dim=args.topic_dim,
                           list_str_y=task.list_str_y,
                           list_d_tr=task.list_domain_tr,
                           gamma_d=args.gamma_d,
                           gamma_y=args.gamma_y,
                           beta_t=args.beta_t,
                           beta_x=args.beta_x,
                           beta_y=args.beta_y,
                           beta_d=args.beta_d)

        model = ModelWrapMatchDGLogit(model, list_str_y=task.list_str_y)
        return model

    def fun_build_alex_ctr():
        return AlexNetNoLastLayer(flag_pretrain=True)

    if task.isize.h > 100:  # FIXME
        return fun_build_alex_ctr, fun_build_alex_erm
    raise NotImplementedError

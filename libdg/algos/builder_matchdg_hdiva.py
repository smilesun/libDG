from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.trainers.train_matchdg import TrainerMatchDG
from libdg.models.model_diva import ModelDIVA
from libdg.models.wrapper_matchdg import ModelWrapMatchDGLogit
from libdg.utils.utils_cuda import get_device
from libdg.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
from libdg.compos.pcr.request import RequestVAEBuilderCHW
from libdg.algos.observers.c_obvisitor_gen import ObVisitorGen
from libdg.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp




class NodeAlgoBuilderMatchDGDIVA(NodeAlgoBuilder):
    """
    Instead of using an ERM model, use DIVA as feature extraction
    """
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)
        fun_build_ctr, fun_build_erm_phi = \
            get_ctr_model_erm_creator(args, task)
        model = fun_build_erm_phi()
        model = model.to(device)
        ctr_model = fun_build_ctr()
        ctr_model = ctr_model.to(device)
        observer = ObVisitor(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)
        trainer = TrainerMatchDG(exp, task, ctr_model, model, observer, args, device)
        # model = ModelWrapMatchDGLogit(model, list_str_y=list_str_y)
        return trainer


def get_ctr_model_erm_creator(args, task):
    def fun_build_alex_erm():
        request = RequestVAEBuilderCHW(task.isize.c, task.isize.h, task.isize.w)
        node = VAEChainNodeGetter(request)()
        model = ModelDIVA(node,
                          zd_dim=args.zd_dim, zy_dim=args.zy_dim,
                          zx_dim=args.zx_dim,
                          list_str_y=task.list_str_y,
                          list_d_tr=task.list_domain_tr,
                          gamma_d=args.gamma_d,
                          gamma_y=args.gamma_y,
                          beta_x=args.beta_x,
                          beta_y=args.beta_y,
                          beta_d=args.beta_d)
        return model

    def fun_build_alex_ctr():
        """
        Constrastive learning
        """
        request = RequestVAEBuilderCHW(task.isize.c, task.isize.h, task.isize.w)
        node = VAEChainNodeGetter(request)()
        model = ModelDIVA(node,
                          zd_dim=args.zd_dim, zy_dim=args.zy_dim,
                          zx_dim=args.zx_dim,
                          list_str_y=task.list_str_y,
                          list_d_tr=task.list_domain_tr,
                          gamma_d=args.gamma_d,
                          gamma_y=args.gamma_y,
                          beta_x=args.beta_x,
                          beta_y=args.beta_y,
                          beta_d=args.beta_d)
        return model

    if task.isize.h > 100:
        return fun_build_alex_ctr, fun_build_alex_erm
    raise NotImplementedError

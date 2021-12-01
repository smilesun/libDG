"""
When class names and numbers does not match across different domains
"""
from libdg.tasks.task_folder import NodeTaskFolderClassNaMismatch

def mk_task_folder(extensions,
                   list_str_y,
                   dict_domain_folder_name2class,
                   dict_domain_img_trans,
                   img_trans_te,
                   isize,
                   dict_domain2imgroot,
                   taskna,
                   succ=None):
    """
    task maker: e.g.
    """
    class NodeTaskFolderDummy(NodeTaskFolderClassNaMismatch):
        @property
        def task_name(self):
            """
            The basic name of the task, without configurations
            """
            return taskna

        def conf(self, args):
            self.extensions = extensions
            self.list_str_y = list_str_y
            self._dict_domain_folder_name2class = dict_domain_folder_name2class
            self.isize = isize
            self.set_list_domains(list(self._dict_domain_folder_name2class.keys()))
            self.dict_domain2imgroot = dict_domain2imgroot
            self._dict_domain_img_trans = dict_domain_img_trans
            self.img_trans_te = img_trans_te

        def init_business(self, args):
            self.conf(args)
            super().init_business(args)
    return NodeTaskFolderDummy(succ=succ)


def test_fun():
    from libdg.utils.arg_parser import mk_parser_main
    from libdg.tasks.utils_task import ImSize
    from torchvision import transforms
    # from libdg.tasks.utils_task import img_loader2dir
    # import os

    node = mk_task_folder(extensions={"caltech": "jpg", "sun": "jpg", "labelme": "jpg"},
                          list_str_y=["chair", "car"],
                          dict_domain_folder_name2class={
                              "caltech": {"auto": "car", "stuhl": "chair"},
                              "sun": {"viehcle": "car", "sofa": "chair"},
                              "labelme": {"drive": "car", "sit": "chair"}
                          },
                          dict_domain_img_trans={
                              "caltech": transforms.Compose([transforms.Resize((224, 224)), ]),
                              "sun": transforms.Compose([transforms.Resize((224, 224)), ]),
                              "labelme": transforms.Compose([transforms.Resize((224, 224)), ]),
                          },
                          isize=ImSize(3, 224, 224),
                          dict_domain2imgroot={
                              "caltech": "zdpath/vlcs_small_class_mismatch/caltech/",
                              "sun": "zdpath/vlcs_small_class_mismatch/sun/",
                              "labelme": "zdpath/vlcs_small_class_mismatch/labelme/"},
                          taskna="mini_vlcs")

    parser = mk_parser_main()
    # batchsize bs=2 ensures it works on small dataset
    args = parser.parse_args(["--te_d", "1", "--bs", "2"])
    node.init_business(args)
    node.get_list_domains()
    print(node.list_str_y)
    print(node.list_domain_tr)
    print(node.task_name)
    node.sample_sav(args.out)
    # alternatively:
    # folder_na = os.path.join(args.out, "task_sample", node.task_name)
    # img_loader2dir(node.loader_te,
    #               list_domain_na=node.get_list_domains(),
    #               list_class_na=node.list_str_y,
    #               folder=folder_na,
    #               batches=10)

    # img_loader2dir(node.loader_tr,
    #               list_domain_na=node.get_list_domains(),
    #               list_class_na=node.list_str_y,
    #               folder=folder_na,
    #               batches=10)

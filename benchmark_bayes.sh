python main_out.py --te_d=0 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/B_net_small.py --epos=500
python main_out.py --te_d=0 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/F_net_small.py --epos=500
python main_out.py --te_d=0 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/Bayesian3Conv3FC.py --epos=500
python main_out.py --te_d=0 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/F3Conv3FC.py --epos=500

# does domain generlaization algorithm works better than Bayes net?

python main_out.py --te_d=0 --aname=diva --task=mnistcolor4 --epos=500

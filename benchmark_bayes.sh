# The bayes version
python main_out.py --te_d=0 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/B_net_small.py --epos=500
# The frequenst version, the same architecture
python main_out.py --te_d=31 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/F_net_small.py --epos=500

# Now we do not use batchnorm
# The bayes version
python main_out.py --te_d=0 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/Bayesian3Conv3FC.py --epos=500
# The frequenst version, the same architecture
python main_out.py --te_d=0 --aname=deepall --task=mnistcolor4 --mpath=./libdg/models/bnn/F3Conv3FC.py --epos=500

# does domain generlaization algorithm works better than Bayes net?

python main_out.py --te_d=0 --aname=diva --task=mnistcolor4 --epos=500


# Reference
#1. Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization∗
# https://arxiv.org/pdf/2101.09436.pdf
#2.Variational Resampling Based Assessment of Deep Neural Networks under Distribution Shift
# https://arxiv.org/pdf/1906.02972.pdf

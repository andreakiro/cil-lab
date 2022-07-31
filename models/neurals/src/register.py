# Register for models and datasets
##################################

from easydict import EasyDict as edict

from src.models.deeprec import AutoEncoder
from src.proc.training.deeprec import train_deeprec
from src.proc.testing.deeprec import eval

from src.models.lightgcn import LightGCN
from src.proc.training.lightgcn import train_lightgcn
from src.proc.testing.lightgcn import test_lightgcn

register = edict()
unavailable = lambda x: x

# MODEL REGISTER

register.deeprec = edict()
register.deeprec.model = AutoEncoder
register.deeprec.train = train_deeprec
register.deeprec.test = eval

register.lightgcn = edict()
register.lightgcn.model = LightGCN
register.lightgcn.train = train_lightgcn
register.lightgcn.test = test_lightgcn

# DATA REGISTER

register.datapaths = edict()
register.datapaths.cil = '../../../data/data_train.csv'
register.datapaths.cil_sample = '../../../data/sampleSubmission.csv'
# PATH TO PROJECT MAIN DATA
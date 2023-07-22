

import numpy as np
import pandas as pd
import random 
from math import log
from keras.utils.np_utils import to_categorical

from keras.models import load_model
from MyEncode import blo_encode_920
from Closest_pep_net import closest_pep_net,mhc_net
from Net_Dict_all import MHC_Net_Dict,Pep_Net_Dict
import sys

###############Read file name ###############
file = sys.argv[1]

######################### binding model ##############

pep_length = 9
f = lambda x: len(x)
mhci = pd.read_csv('../data/binding_train.csv')
mhci['peptide_length'] = mhci.sequence.apply(f)
mhci = mhci[mhci.peptide_length == pep_length]

pep_net_dict = np.load('../dict/bind_pep.npy',allow_pickle=True).item()
mhc_net_dict = np.load('../dict/bind_hla.npy',allow_pickle = True).item()
mhc_median_network = np.load('../dict/bind_mhc_network.npy')
mhc_median_network =[float(value) for value in mhc_median_network[:4]];mhc_median_network.append('median')

######prediction on sample data##########

#mhci_response = pd.read_csv('../data/sample.txt',sep = '\t')
mhci_response = pd.read_table(file)

mhci_response['peptide_length'] = mhci_response.sequence.apply(f)
mhci_response = mhci_response[mhci_response.peptide_length == pep_length].reset_index(drop = True)

peptide = {}
network_pep = {}
network_mhc = {}
for pep in list(set(mhci_response.sequence)):
    peptide[pep] = blo_encode_920(pep)
    network_pep[pep] = closest_pep_net(pep,pep_net_dict)


for mhc in list(set(mhci_response.mhc)):
    network_mhc[mhc] = mhc_net(mhc,mhc_median_network,mhc_net_dict)

network = []
peptides = []
pep_dist = []
mhc_class = []
for pep,mhc in zip(mhci_response.sequence,mhci_response.mhc):
    network.append(np.array(network_pep[pep][:4]+network_mhc[mhc][:4]).reshape(8,1))
    pep_dist.append(network_pep[pep][4])
    peptides.append(peptide[pep])
    mhc_class.append(network_mhc[mhc][4])


model_bind = load_model('model/bind.h5')
# 预测
prediction = model_bind.predict([np.array(peptides),np.array(network)])

# 写入结果
result = pd.DataFrame({'mhc':mhci_response.mhc,'sequence':mhci_response.sequence,\
                   'pred_affinity':prediction.flatten()})

# print(result)

################## immunogenic model ########################

mhci = pd.read_csv('../data/immunogenic_train.csv',index_col = 0)
mhci['peptide_length'] = mhci.sequence.apply(f)
mhci = mhci[mhci.peptide_length == pep_length]
immunity_bin=[1 if value ==1 else 0 for value in mhci['Label']]
categorical_labels = to_categorical(immunity_bin, num_classes=None)


pep_net_dict = np.load('../dict/immuno_pep.npy',allow_pickle=True).item()
mhc_net_dict = np.load('../dict/immuno_hla.npy',allow_pickle = True).item()
mhc_median_network = np.load('../dict/immuno_mhc_network.npy')
mhc_median_network =[float(value) for value in mhc_median_network[:4]];mhc_median_network.append('median')

# 构造免疫性模型的特征数据
peptide = {}
network_pep = {}
network_mhc = {}
for pep in list(set(mhci_response.sequence)):
    peptide[pep] = blo_encode_920(pep)
    network_pep[pep] = closest_pep_net(pep,pep_net_dict)

for mhc in list(set(mhci_response.mhc)):
    network_mhc[mhc] = mhc_net(mhc,mhc_median_network,mhc_net_dict)

network = []
peptides = []
pep_dist = []
mhc_class = []
for pep,mhc in zip(mhci_response.sequence,mhci_response.mhc):
    network.append(np.array(network_pep[pep][:4]+network_mhc[mhc][:4]).reshape(8,1))
    pep_dist.append(network_pep[pep][4])
    peptides.append(peptide[pep])
    mhc_class.append(network_mhc[mhc][4])

# 加载模型，预测，写入结果
model_immuno = load_model('model/immuno.h5')

prediction =np.argmax(model_immuno.predict([np.array(peptides),np.array(network)]),axis=1)
probability = [value[1] for value in model_immuno.predict([np.array(peptides),np.array(network)])]

result['pred_immuno'] = prediction.flatten()
result['immuno_probability'] = probability


# 保存结果
result.to_csv('result_prediction.txt',sep = '\t',index=False)


import numpy as np
import pandas as pd

def MHC_Net_Dict(mhci):
    mhc_net_dict = {}
    tmp = mhci[['mhc','hla_degree', 'hla_close', 'hla_between', 'hla_evcent']].drop_duplicates()
    tmp.set_index(['mhc'],inplace = True)
    for mhc in list(set(mhci.mhc)):
        if mhc not in mhc_net_dict:
            mhc_net_dict[mhc]=tmp.loc[mhc]
    return (mhc_net_dict)


def Pep_Net_Dict(mhci):
    tmp = mhci[['sequence','pep_degree', 'pep_close', 'pep_between', 'pep_evcent']].drop_duplicates()
    tmp.set_index(['sequence'],inplace = True)
    pep_net_dict = {}
    for pep in list(set(mhci.sequence)):
        if pep not in pep_net_dict:
            pep_net_dict[pep]=tmp.loc[pep]
    return pep_net_dict
    
mhci = pd.read_csv('../data/immuno_all.csv')
# print(mhci)
res = MHC_Net_Dict(mhci)
np.save("../dict/immuno_hla.npy", res)

res = Pep_Net_Dict(mhci)
np.save("../dict/immuno_pep.npy", res)

mhci = pd.read_csv('../train/bind_all.csv')
res = MHC_Net_Dict(mhci)
np.save("../dict/bind_hla.npy", res)

res = Pep_Net_Dict(mhci)
np.save("../dict/bind_pep.npy", res)

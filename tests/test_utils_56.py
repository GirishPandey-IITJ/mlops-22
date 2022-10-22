import sys
sys.path.append('..')
sys.path.append('.')
from utils import getall_h_params_comb

def test_get_pram():
    g_ls = [0.2,0.09,0.05,0.025]
    c_ls = [0.2,0.5,0.9,1,1.5]

    params = {}
    params['gamma'] = g_ls
    params['C'] = c_ls
    h_pcomb = getall_h_params_comb(params)
    assert len(h_pcomb) == len(g_ls)*len(c_ls)
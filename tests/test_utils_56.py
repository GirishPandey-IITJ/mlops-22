import sys
sys.path.append('..')
sys.path.append('.')
from utils import getall_h_params_comb,distinct_op

def test_get_pram():
    g_ls = [0.2,0.09,0.05,0.025]
    c_ls = [0.2,0.5,0.9,1,1.5]

    params = {}
    params['gamma'] = g_ls
    params['C'] = c_ls
    h_pcomb = getall_h_params_comb(params)
    assert len(h_pcomb) == len(g_ls)*len(c_ls)

def test_baised():
    l = 1
    k = {0,1,2,3,4,5,6,7,8,9}
    baised_output_set = distinct_op(k)
    assert len(baised_output_set) != 1


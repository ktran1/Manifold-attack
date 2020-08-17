# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:05:43 2020

@author: Khanh-Hung TRAN
khanhhung92vt@gmail.com
"""

def prox_positive_and_sum_constraint_2d(x,c):
    """ 
    Projection to get all element of x are positive and sum of all element of x is c
    Used in splitting method
    """

    n = x.size()[1]
    k = (c - torch.sum(x,dim=1))/float(n)
    x_0 = x + k[:,None]
    while len(torch.where(x_0 < 0)[0]) != 0:
        idx_negative = torch.where(x_0 < 0)        
        x_0[idx_negative] = 0.

        one = x_0 > 0
        n_0 = one.sum(dim=1)
        k_0 =(c - torch.sum(x_0,dim =1))/ n_0
        x_0 = x_0 + k_0[:,None] * one
                
    return x_0


def prox_positive_sum_constraint_2d_and_first_element(x,c,tau):
    """ 
    Projection to get all element of x are positive and sum of all element of x is c
    The first element is always >= tau (tau < c)
    Used in splitting method
    """
    x0 = x.clone()
    x0[:,0] = x0[:,0] - tau
    x_p = prox_positive_and_sum_constraint_2d(x0,c-tau)
    x_p[:,0] = x_p[:,0] + tau
    
    return x_p
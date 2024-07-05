from argparse import ZERO_OR_MORE
from ast import Mod
from distutils.command.config import config
from logging import error
from re import A
#from cv2 import determinant
from matplotlib import markers
import matplotlib.patches as patches
import numpy as np
import itertools
import pandas as  pd
import scipy.linalg as la
import scipy.integrate as inte
import scipy.optimize as opt
import scipy.sparse as sparse

import scipy.special as sp

from astropy.io import ascii

import ast

from pfapack import pfaffian as pf

from tabulate import tabulate

import mpmath as mpm

import collections as coll

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import colorcet as cc


import  IPython.display as IPd
import os
import pathlib
import csv
import io

from datetime import datetime

try:
    import cmasher as cmr
except ImportError:
    print('Can Get cmasher')

try:
     import seaborn as sns
except ImportError:
    print('Can Get seaborn')



import time

#plt.rcParams['text.usetex'] = True

def c_eff(b):
    if b == 0:
        return 0
    t = np.sin(2*np.arctan(1/b))
    at = np.abs(t)

    ceff = at/2 - 1/2
    ceff += -3/(np.pi**2)*((at+1)*np.log(at+1)*np.log(at)+(at-1)*sp.spence(at)+(at+1)*sp.spence(1+at))

    return ceff

def Entropy_From_H(H, Which = 'Mod Ham', tol = 10e-10, Return_s_arr = False):
    if Which == 'Eigs':
        eigs = np.real(H)
    
    if Which == 'Weights':
        H = Ham_From_Weights(H) 
        eigs = la.eigh(H)[0]

    if Which == 'Mod Ham':
        eigs = np.real(la.eigh(H)[0])

    eigs = eigs[eigs>tol]
    s_arr = np.log(1+np.e**(-eigs))+np.e**(-eigs) / (1+np.e**(-eigs))*eigs

    if Return_s_arr:
        return s_arr
    
    s_arr = s_arr[eigs>tol]

    return(np.sum(s_arr))

# def PrintMatrix(mat, Round = 10, buffer = 3):
#     iscomplex = False
#     isstring = False
#     if np.iscomplexobj(mat):
#         iscomplex = True
#         toprint = np.around(mat, Round)
#     if type(mat[0][0]) == str or type(mat[0][0]) == np.str_:
#         isstring = True
#         toprint = mat
#     else:
#         if not isstring:
#             toprint = np.around(mat,Round)
#     maxlength = max(len(str(entree)) for row in toprint for entree in row) + buffer
#     colmaker = "{:{width}}"*len(toprint[0])
#     for row in toprint:
#         row_to_print = []
#         if not isstring:
#             for i in row:
#                 if i == 0:
#                     i = 0
#                 if iscomplex and np.real(i) != 0 and np.imag(i) == 0:
#                     i = np.around(np.real(i),Round)
#                 if iscomplex and np.real(i) == 0 and np.imag(i) != 0:
#                     i = np.around(np.imag(i)*1j,Round)
#                 row_to_print.append(str(i))
#         if isstring:
#             row_to_print = row
#         print(colmaker.format(*row_to_print,width=maxlength))
#     print("")

def Ham_From_Weights(w):
    mod_ham = np.zeros((len(w)+1,len(w)+1), dtype = type(w[0]))
    for a in range(len(w)):
        mod_ham[a,a+1] = w[a]
        mod_ham[a+1,a] = -mod_ham[a,a+1]
    return mod_ham

def PrintMatrix(mat, Round = 10, buffer = 3, Return = False):
    if type(mat) == sparse._csr.csr_matrix:
        mat = mat.toarray()
    if not isinstance(mat[0,0], str):
        columns = np.round(mat,Round).astype(str)
        columns[columns.astype(complex)==0] = '0'
    else:
        columns = mat.astype(str)
    
    maxlength = max(len(str(entree)) for row in columns for entree in row)
    headers = [' '*(maxlength//2+buffer)]*len(columns)
    table = tabulate(columns, tablefmt='plain', headers=headers ,numalign='center')
    
    if Return:
        return table
    print(table)

def PrintEigensystem(mat, Round = 10, hermitian = False, Return = False, ScientificNotation = True, ScientificNotation_Decimals = 1):
    if hermitian:
        eigensystem = la.eigh(mat)
    else:
        eigensystem = la.eig(mat)
    columns = np.round(eigensystem[1],Round).astype(str)
    
    if not ScientificNotation:
        headers =  np.round(eigensystem[0],Round)
    if ScientificNotation:
        headers = []
        for num in eigensystem[0]:
            headers.append(str("{:."+str(ScientificNotation_Decimals)+"e}").format(num))
        headers = np.array(headers)
    table = tabulate(columns, headers=headers)
    print(table)
    if Return:
        return eigensystem
    
def CartesianProduct2D(x_list,y_list):
    res = []
    for x in x_list:
        for y in y_list:
            res.append([x,y])
    return res

def listZip(x_list,y_list):
    if len(x_list) != len(y_list):
        print("Need Both Lists To Be Same Length")
        return None
    res = []
    for i in range(len(x_list)):
        res.append([x_list[i],y_list[i]])

    return res

sigma_x = np.array([[0,1],[1,0]], dtype=complex)
sigma_y = np.array([[0,-1j],[1j, 0]], dtype=complex)
sigma_z = np.array([[1,0],[0,-1]], dtype=complex)
Id = np.array([[1,0],[0,1]], dtype=complex)

def JordanWignerX(j, N):
    matrices = []
    for k in range(j-1):
        matrices.append(-sigma_z)
    matrices.append(sigma_x)
    for k in np.arange(j+1, N-1):
        matrices.append(Id)
    return Multi_Kronecker(matrices)

def JordanWignerY(j, N):
    matrices = []
    for k in range(j-1):
        matrices.append(-sigma_z)
    matrices.append(sigma_y)
    for k in np.arange(j+1, N-1):
        matrices.append(Id)
    return Multi_Kronecker(matrices)

def JordanWignerP(j, N):
    matrices = []
    for k in range(j-1):
        matrices.append(-sigma_z)
    matrices.append((sigma_x + 1j*sigma_y)/2.)
    for k in np.arange(j+1, N-1):
        matrices.append(Id)
    return Multi_Kronecker(matrices)

def JordanWignerM(j, N):
    matrices = []
    for k in range(j-1):
        matrices.append(-sigma_z)
    matrices.append((sigma_x - 1j*sigma_y)/2.)
    for k in np.arange(j+1, N-1):
        matrices.append(Id)
    return Multi_Kronecker(matrices)

def JordanWignerMajorana(m,N):
    if m%2 == 0:
        return JordanWignerX(m,2*N)
    if m%2 == 1:
        return JordanWignerY(m, 2*N)

def Multi_Kronecker(matrices, Sparse = False):
    '''If sparse, the format is csr.'''
    a = matrices[0]
    if not Sparse:
        for mat in matrices[1:]:
            a = np.kron(a, mat)
    if Sparse:
        for mat in matrices[1:]:
            a = sparse.kron(a, mat, format = 'csr')
            a.eliminate_zeros()
    return a

def Single_Site_Operator(op, Id, m, N, Sparse = False):
    #m = m-1
    matrices = [Id]*N
    matrices[m] = op
    return Multi_Kronecker(matrices, Sparse = Sparse)

def Expectation_Value(state, op, Round = 10):
    a = np.round(op @ state, Round)
    b = np.round(np.conj(state) @ a, Round)
    return b

def Spin_Charge_Operator(N, Sparse = False):
    return Multi_Kronecker([sigma_z]*N, Sparse = Sparse)

def Matrix_Log(mat, Debug = False, hermitian = False):
    if not hermitian:
        eigenvalues, eigenvectors = la.eig(mat)
    if hermitian:
        eigenvalues, eigenvectors = la.eigh(mat)
        eigenvalues = eigenvalues.astype(complex)
    lam = np.diag(np.log(eigenvalues)).astype(complex)
    if Debug:
        PrintMatrix(lam)
    return eigenvectors@lam@np.conj(eigenvectors.T)

def Matrix_Log_SVD(mat, Debug = False):
    U, s, Vd = la.svd(mat)
    lam = np.diag(np.log(s)).astype(complex)
    if Debug:
        PrintMatrix(lam)
    return U@lam@Vd

def Periodic_Beta(x,L,N):
    return 4*N*np.sin(np.pi*(x)/N)*np.sin(np.pi*(L-x)/N) / (2*np.sin(np.pi*L/N)) if L!=N else 0

def Open_On_Boundary_Beta(x,L,N):
    return 4*N*(np.cos(np.pi*(x)/(N)) - np.cos(np.pi*(L)/(N)))/(2*np.sin(np.pi*(L)/(N)))

def MPM_Periodic_Beta(x,L,N):
    return mpm.sin(mpm.pi*(x)/N)*mpm.sin(mpm.pi*(L-x)/N) / (2*mpm.sin(mpm.pi*L/N))*4*N if L!=N else 0

def MPM_Open_On_Boundary_Beta(x,L,N):
    return 4*N*(mpm.cos(mpm.pi*(x)/(N)) - mpm.cos(mpm.pi*(L)/(N)))/(2*mpm.sin(mpm.pi*(L)/(N)))

def Get_Ham_From_Ops_And_Weights(operators, w, size, dtype = complex):
    mod_ham = np.zeros((size, size), dtype=dtype)
    for i in range(len(operators)):
            op = operators[i]
            if isinstance(op[0], (list, np.ndarray)):
                for opp in op:            
                    mod_ham[opp[0],opp[1]] = w[i]
                    mod_ham[opp[1],opp[0]] = -mod_ham[opp[0],opp[1]]
            else:
                mod_ham[op[0],op[1]] = w[i]
                mod_ham[op[1],op[0]] = -mod_ham[op[0],op[1]]
    
    return mod_ham
class MajoranaFermionChain:
    # h_i = J σx_i σx_i+1 + g σz_i
    # h_i = J a_2l a_{2l+1} + ga_{2l-1}a_{2l}
    def __init__(self, N = 100, Jx = 1, Jy = 0, g = 1, b_sigma = 0, Q = 1, boundaries = "periodic", cutoff = 14, load_name = None, load_results_only = False, data_directory = "data/", use_f64=False):
        self.data_directory = data_directory
        self.load_results_only = load_results_only
        self.N = N
        self.Jx = Jx
        self.Jy = Jy
        self.g = g
        self.boundaries = boundaries
        self.cutoff = cutoff
        self.b_sigma = b_sigma # Topological Defect Term: EQ 2 of https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.128.090603
        self.Q = Q
        

        if type(load_name) == str:
            self.load(load_name, directory = self.data_directory, load_results_only = self.load_results_only)
        
        
        if isinstance(self.Jx, (np.floating, float, int)):
            self.Jx = self.Jx * np.ones(self.N)
        
        if isinstance(self.Jy, (np.floating, float, int)):
            self.Jy = self.Jy * np.ones(self.N)
        
        if isinstance(self.g, (np.floating, float, int)):
            self.g = self.g * np.ones(self.N)
        
        if isinstance(self.b_sigma, (np.floating, float, int)):
            self.b_sigma = self.b_sigma * np.ones(self.N)
        
        if not isinstance(self.Jx, np.ndarray):
            print(f"For Jx, Please uses either a single constant or a dim({str(self.N)}) array")
            return None
        if not isinstance(self.Jy, np.ndarray):
            print(f"For Jy, Please uses either a single constant or a dim({str(self.N)}) array")
            return None
        if len(self.Jx) != self.N:
            print("Jx Array is length "+str(len(self.Jx))+". Please use length "+str(self.N))
            return None

        if len(self.Jy) != self.N:
            print("Jy Array is length "+str(len(self.Jy))+". Please use length "+str(self.N))
            return None

        if not isinstance(self.g, np.ndarray):
            print("Please uses either a single constant or a dim("+str(self.N)+") array")
            return None
        if len(self.g) != self.N:
            print("Array is length "+str(len(self.g))+". Please use length "+str(self.N))
            return None

        if not isinstance(self.b_sigma, np.ndarray):
            print("Please uses either a single constant or a dim("+str(self.N)+") array")
            return None
        if len(self.b_sigma) != self.N:
            print("Array is length "+str(len(self.b_sigma))+". Please use length "+str(self.N))
            return None

        self.Ham = None
        self.Ham_Eigensystem = None
        self.Ground_State_Energy = None
        self.Ground_State = {}
        self.Ground_State['empty'] = None
        self.Ground_State['filled'] = None
        self.Gamma_A = {}
        self.Gamma_A['empty'] = None
        self.Gamma_A['filled'] = None
        self.Gamma_A['flipped'] = None
        self.Gamma_A['both'] = None
        self.B = None
        self.c_Fit = None
        self.Num_for_Fit = None
        self.Ground_State_Fermion_Number = None
    


        # EE = Entanglement Entropy
        # ES = Entanglement Spectrum
        # FLN = Fermionic Logarithmic Negativity
        # BLN = Bosonic Logarithmic Negativity
        # GAPRE = Gamma_A_Partial_ReEigenvalues
        # GAPTE = Gamma_A_Partial partially transposed eigenvalues
        self.columns = ["GAPRE", "GAPTE", "ES", "EE", "FLN", "BLN"]
        self.model_DF = pd.DataFrame(columns = self.columns, dtype=object)

        

        self.ExpectationValues = {}
       
    def Ham_Builder(self, debug = False):
        if isinstance(self.Ham_Eigensystem, np.ndarray):
            return None
        
        NN = 2*self.N
        A = np.zeros((NN, NN), dtype = np.float64)

        epsilon = lambda m,n: -1 if m>n else 1

        limit = None
        if self.boundaries == 'periodic':
            limit = self.N

        if self.boundaries == 'open':
            limit = self.N-1
        if limit == None:
            print(f' Incorrect boundary condition of "{self.boundaries}" given.')
            return None

        if debug:
            print(f'Boundary conditions: {self.boundaries}\n    N is {self.N}, "limit" is {limit-1} (index starts at 0)\n')

        if debug:
            print(f'Doing XX and YY terms')
        for l in range(limit):
            if debug:
                print(f"    l = {l}")

            # XX Interaction    
            m = (2*l+1) % NN
            n = (2*l+2) % NN
            A[m,n] = self.Jx[l]*epsilon(m,n)
            A[n,m] = -A[m,n]
            if l == self.N-1:
                A[m,n]*=self.Q
                A[n,m]*=self.Q
            if debug:
                print(f'        XX: A[{m},{n}] = {A[m,n]}')
            

            # YY Interaction: Note this has an extra relative minus sign compared to XX
            m = (2*l) % NN
            n = (2*l+3) % NN
            A[m,n] = -self.Jy[l]*epsilon(m,n)
            A[n,m] = -A[m,n]
            if l == self.N-1:
                A[m,n]*=self.Q
                A[n,m]*=self.Q
            if debug:
                print(f'        YY: A[{m},{n}] = {A[m,n]}')

            m = (2*l+1) % NN
            n = (2*l+3) % NN
            A[m,n] += self.b_sigma[l]
            A[n,m] += -self.b_sigma[l]
        
        if debug:
            print('\nDoing Z Terms')
        for l in range(self.N):
            if debug:
                print(f"    l = {l}") #This always goes from 0 to N-1 as it is a term at each site with no hopping. It is not affected by boundary conditions.")
            m = (2*l) % NN
            n = 2*l+1 % NN
            A[m,n] = self.g[l]*epsilon(m,n)
            A[n,m] = -A[m,n]
            if debug:
                print(f'        Z: A[{m},{n}] = {A[m,n]}')
 
       
        self.Ham = -1j/2. * A

    def Get_Ham_Eigensystem(self):
        if isinstance(self.Ham_Eigensystem, np.ndarray):
            return None

        if isinstance(self.Ham, np.ndarray):
            self.Ham_Eigensystem = la.eigh(self.Ham)
        else:
            self.Ham_Builder()
            self.Ham_Eigensystem = la.eigh(self.Ham)

        self.Ground_State_Fermion_Number = len(self.Ham_Eigensystem[0][self.Ham_Eigensystem[0]<0])
    
    def Get_Ground_State_Energy(self, Return = True, zero_modes = 'filled'):
        if self.Ground_State_Energy is not None and self.Ground_State[zero_modes] is not None:
            if Return:
                return self.Ground_State_Energy
            else:
                return None

        if not isinstance(self.Ham_Eigensystem, np.ndarray):
            self.Get_Ham_Eigensystem()
        if zero_modes == 'empty':
            Ground_State = np.round(np.sum(self.Ham_Eigensystem[1][:,np.round(self.Ham_Eigensystem[0], self.cutoff)<0], axis = 1),self.cutoff)
            self.Ground_State['empty'] = Ground_State/np.sqrt(np.sum(Ground_State*np.conj(Ground_State)))
        if zero_modes == 'filled':
            Ground_State = np.round(np.sum(self.Ham_Eigensystem[1][:,np.round(self.Ham_Eigensystem[0], self.cutoff)<=0], axis = 1),self.cutoff)
            self.Ground_State['filled'] = Ground_State/np.sqrt(np.sum(Ground_State*np.conj(Ground_State)))

        self.Ground_State_Energy = np.round(np.sum(self.Ham_Eigensystem[0][self.Ham_Eigensystem[0]<0]),self.cutoff)

        if Return:
            return self.Ground_State_Energy

    def Get_Gamma_A(self, Return = True, debug = False, Full_Debug = False, redo = False, large_Sort_warning = 1000, debug_round = 4, zero_modes = 'filled', Given_Schur = None):
        if (self.Gamma_A[zero_modes] is not None) and not redo:
            if Return:
                return self.Gamma_A[zero_modes]
            else:
                return None

        if self.Ham is None:
            self.Ham_Builder()
        
        A = -2j*self.Ham
        
        if Given_Schur is None:
            B,  W = la.schur(np.real(A))  # A = WBW.T -> GA = W GB W.T
        else:
            B = Given_Schur[0]
            W = Given_Schur[1]

        B = np.round(B, self.cutoff)
        W = np.round(W, self.cutoff)
        if debug and Full_Debug:
            print(f'Starting A:')
            PrintMatrix(A,Round =debug_round)
            print("Starting B Matrix:")
            PrintMatrix(B, Round = debug_round)
            print(f'Det(B_Start): {la.det(B)}')
            print("Starting W Matrix:")
            PrintMatrix(W, Round = debug_round)
            print(f'Det(W_Starting): {la.det(W)}')
        
        Sorted = False
        zero_columns = []
        for i in range(len(B)):
            if np.all(np.round(B[:,i], self.cutoff)==0):
                zero_columns.append(i)
        if len(zero_columns) == 0:
            if debug:
                print("No zero columns.")
            Sorted = True
        
        if len(zero_columns)!= 0:
            if debug: print(f'{len(zero_columns)} Zero Columns')
    
        if not Sorted:
            zero_columns = np.array(zero_columns)
            if debug:
                    print(f'    Zero columns: {zero_columns}')

            zero_columns_goal = []
            for i in range(len(zero_columns)):
                zero_columns_goal.append(self.N - len(zero_columns)//2+i)
            
            for i in range(len(zero_columns)):
                column = zero_columns[i]
                if debug:
                    print(f'On {i}th zero column: column # {column}')
                    print(f'Want this column to be at {zero_columns_goal[0]}')
                if column != zero_columns_goal[i]:
                    n = (zero_columns_goal[i]-column)//2
                    if debug:
                        print(f'Need {n} steps:')
                    for m in np.arange(1,np.abs(n)+1):
                        shift = np.identity(2*self.N)
                        if debug:
                            print(f'    m = {m}')
                        M = column + np.sign(n)*m-1 + np.sign(n)*((m-1))
                        if debug:
                            print(f'step {m}, M = {M}')
                        shift[M,M] = 0
                        shift[M+2,M+2] = 0
                        shift[M+2, M]  = np.sign(n)*1
                        shift[M,M+2] = -np.sign(n)*1

                        B = shift@B@shift.T
                        W= W@shift.T
                        if debug:
                            print(f'Shifting column {column}, on step {m}')
                            if Full_Debug:
                                PrintMatrix(shift, Round = debug_round)
                                PrintMatrix(B, Round = debug_round)
                            print('--------\n')

        if debug:
            print("Flipping Minus Signs")
        signs_flipped = 0
        shift = np.identity(2*self.N)
        for m in range(2*self.N):
            if B[m,m-1] > 0:
                signs_flipped +=1 
                shift[m,m] = 0
                shift[m-1,m-1] = 0
                shift[m-1,m] = 1
                shift[m,m-1] = 1 
        
        B = shift@B@shift.T
        W = W@shift.T

        if debug:
            print(f'{signs_flipped} blocks had minus signs flipped.')
            if Full_Debug: 
                print(f'Det(Shift): {la.det(shift)} (Should be 1)')
                PrintMatrix(shift, Round = debug_round)
                print(f'Final B Matrix:')
                PrintMatrix(B, Round = debug_round)
                print(f'Det(B_Final): {la.det(B)}')
                print(f'Final W Matrix:')
                PrintMatrix(W, Round = debug_round)
                print(f'Det(W_Final): {la.det(W)}')
                print(f'Testing that A - W@B@W.T = 0')
                PrintMatrix(A - W@B@np.conj(W.T), debug_round)
                

        Gamma_B = la.block_diag(*[[[0,1.0],[-1.0,0]]]*self.N)
        if len(zero_columns)!=0:
            if zero_modes != 'filled':
                zero_columns = []
                for i in range(len(B)):
                    if np.all(np.round(B[i],15)==0):
                        zero_columns.append(i)
                
                if debug: print(f'Zero Columns in B: {len(zero_columns)}')
                if zero_modes == 'empty':
                    for i in range(len(zero_columns)//2):
                        if debug: print(f'Removing Zero Mode at Column {i}')
                        Gamma_B[zero_columns[2*i]:zero_columns[2*i+1]+1, zero_columns[2*i]:zero_columns[2*i+1]+1] *= 0 
                #if zero_modes == 'both':
                #    for i in range(len(zero_columns)//2):
                #        #Gamma_B[zero_columns[2*i]:zero_columns[2*i+1]+1, zero_columns[2*i]:zero_columns[2*i+1]+1] = np.abs(Gamma_B[zero_columns[2*i]:zero_columns[2*i+1]+1, zero_columns[2*i]:zero_columns[2*i+1]+1])
                #        Gamma_B[zero_columns[2*i]:zero_columns[2*i+1]+1, zero_columns[2*i]:zero_columns[2*i+1]+1] = np.array([[1j,0],[0,1j]])
                if zero_modes == 'flipped':
                    for i in range(len(zero_columns)//2):
                        Gamma_B[zero_columns[2*i]:zero_columns[2*i+1]+1, zero_columns[2*i]:zero_columns[2*i+1]+1] *= -1
                        

        if debug and Full_Debug:
            print("Gamma_B:")
            PrintMatrix(Gamma_B, Round = debug_round)
            print(f'Det(Gamma_B): {la.det(Gamma_B)}')
        self.B = B
        self.Gamma_A[zero_modes] = np.round(W@(Gamma_B)@W.T,self.cutoff) 
        if debug:
            print(f'Det(Gamma_A) = {la.det(self.Gamma_A[zero_modes])}')

        if Return:
            return self.Gamma_A[zero_modes]

    def Get_Gamma_A_Partial(self, interval, sites = False, debug = False, Redo = False, large_Sort_warning = 1000, debug_round = 4, zero_modes = 'filled', print_mask = False, Full_Debug = False):
        if sites:
            site_list = interval
            interval = []
            for site in site_list:
                interval.append(site)
                interval.append(site)
            
        interval = np.array(interval).astype(int)
        xi_arr = interval[::2]%(self.N)
        xf_arr = interval[1::2]%(self.N)
        if debug:
            print(f' Starting Points {xi_arr}')
            print(f'End Points: {xf_arr}')
        
        if self.Gamma_A[zero_modes] is None:
            self.Get_Gamma_A(Return = False, zero_modes=zero_modes, redo=Redo, large_Sort_warning=large_Sort_warning, debug=debug, Full_Debug=Full_Debug)

        Gamma_A = self.Gamma_A[zero_modes]

        mask = np.zeros((2*self.N,2*self.N), dtype=bool)
        false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
        for i in range(len(xi_arr)):
            xi = xi_arr[i]
            xf = xf_arr[i]
            index_arr = np.indices((2*self.N,2*self.N))
            tf_start = index_arr >= (2*xi)
            if xi >= 0:
                tf_start = ~tf_start
            tf_end = index_arr < 2*(xf+1)
            or_xy = np.logical_xor(tf_start,tf_end)
            mask_i = np.logical_and(or_xy[0],or_xy[1])
            if debug:
                print(f"Mask for interval {i+1}")
                PrintMatrix(np.int8(mask_i))
            mask = np.logical_or(mask, mask_i)
        
        false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
            
        if debug or print_mask:
            print("Mask:")
            PrintMatrix(np.int8(mask))


        Gamma_A_Partial = np.delete(np.delete(Gamma_A,false_loc,axis=0), false_loc, axis = 1)

        return Gamma_A_Partial

    def DensityMatrixChargeParity(self, debug = False, zero_modes = 'filled'):  #https://scipost.org/SciPostPhysLectNotes.54/pdf (eq 13)
        if self.Gamma_A[zero_modes] is None:
            self.Get_Gamma_A(zero_modes=zero_modes)
        Q = np.zeros((2*self.N, 2*self.N), dtype = complex)
        delta = lambda j,k: 1 if j==k  else 0
        exp_am_an = lambda m,n: delta(m,n) + 1j*self.Gamma_A[zero_modes][m, n]
        
        for i in range(len(Q)):
            for j in range(i):
                Q[i,j] = exp_am_an(i,j)
                Q[j,i] = -exp_am_an(i,j)

        if debug: print(f'Unrounded Q: {pf.pfaffian(Q)}')
        return np.round(pf.pfaffian(Q))*(1j)**(2*self.N)
        
    def FermionNumberParity(self, debug = False, zero_modes = 'filled'): 
        if self.Ham_Eigensystem is None:
            self.Get_Ground_State_Energy(Return = False, zero_modes=zero_modes)
        if zero_modes == 'filled' or zero_modes == 'flipped':
            majorana_number = len(self.Ham_Eigensystem[0][np.round(self.Ham_Eigensystem[0], self.cutoff)<=0])
        if zero_modes == 'empty':
            majorana_number = len(self.Ham_Eigensystem[0][np.round(self.Ham_Eigensystem[0], self.cutoff)<0])
        if debug:
            print(f'Number of Majorana Modes: {majorana_number}')
        return int((-1)**(majorana_number/2))
        
    def Entanglement_Entropy(self, interval, Return = True, debug = False, Round = 10, print_mask = False, sites = False, Redo = False, zero_modes = 'filled', entropy_function = 'default', save = True, large_Sort_warning = 1000):
        ''' Note that this entanglement entropy code does not do the empty interval, the interval [a, a] is treated as just the 'a' site. This is why [0,0] returns non-zero, it is the single site entropy.'''
        Bipartite_S = lambda x: -x*np.log(x) - (1-x)*np.log(1-x)
        if entropy_function != 'default':
            save = False
            Redo = True
        if entropy_function == 'default':
            entropy_function = lambda x: Bipartite_S(x)
        if sites:
            site_list = interval
            interval = []
            for site in site_list:
                interval.append(site)
                interval.append(site)

        interval_str = str(interval)+'_'+zero_modes
        interval = np.array(interval).astype(int)
        xi_arr = interval[::2]%(self.N)
        xf_arr = interval[1::2]%(self.N)
        if debug:
            print(f' Starting Points {xi_arr}')
            print(f'End Points: {xf_arr}')
    
        
        if len(xi_arr) != len(xf_arr):
            print("Bad Interval")
            return None

        if interval_str not in self.model_DF.index:
            self.model_DF.loc[interval_str] = np.empty(len(self.model_DF.columns), dtype=object)
        
        if not self.model_DF["EE"][interval_str] == None and not Redo:
            if Return: 
                return self.model_DF["EE"][interval_str] 
            else:
                return None
        
    
        Gamma_A_Partial = self.Get_Gamma_A_Partial(interval, debug = debug, print_mask = print_mask, zero_modes = zero_modes, large_Sort_warning = large_Sort_warning, Redo = Redo)
        Gamma_A = self.Gamma_A[zero_modes]    
                    
        if debug:
            print("Shape of Gamma_A: "+str(Gamma_A.shape))
            print("Shape of Gamma_A_Partial: "+str(Gamma_A_Partial.shape))
            print("DataType = "+str(Gamma_A_Partial.dtype))
            print("Gamma_A_Partial:")
            PrintMatrix(Gamma_A_Partial,Round=4)

        if Gamma_A_Partial.size == 0:
            self.model_DF["EE"][interval_str] = 0
            self.model_DF["GAPRE"][interval_str] = np.array([0])
        
        else: 
            
            ReEigs = np.round(la.eigh(-1j*Gamma_A_Partial)[0],self.cutoff)
            
            self.model_DF["GAPRE"][interval_str] = ReEigs
            
            if debug:
                print("Eigenvalues: "+str(ReEigs))
            x_arr = (1+np.abs(ReEigs))/2      
            if debug:
                print("Nus: "+str(x_arr))
            
            Sl_arr = entropy_function(x_arr)
            
            for i in range(len(Sl_arr)):
                if np.isnan(Sl_arr[i]):
                    Sl_arr[i] = 0
            
            if debug:
                print(f'Sl_arr:\n{Sl_arr}')
                
            S = np.round(np.sum(Sl_arr)/2,Round)
            
            if save:
                self.model_DF["EE"][interval_str] = S
        
        if Return:
            return S

    def Get_Entanglement_Entropies(self, intervals, Return = False, Print = False, sites = False, zero_modes = 'filled'):
        results = []
        for interval in intervals:
            if Print:
                print("Doing Entropy Interval = "+str(interval)+"\n")
            results.append(self.Entanglement_Entropy(interval, Return=True, sites = sites, zero_modes=zero_modes))
            if Print:
                IPd.clear_output()
        if Return:
            return np.array(results)

    def Entanglement_Spectrum(self, interval, max_number_of_eigenvalues = 100, Return = False, sites = False, zero_modes = 'filled', debug = False):
        if sites:
            site_list = interval
            interval = []
            for site in site_list:
                interval.append(site)
                interval.append(site)

        interval_str = str(interval)+'_'+zero_modes

        if interval_str not in self.model_DF.index:
            self.model_DF.loc[interval_str] = np.empty(len(self.model_DF.columns), dtype=object)
        
        if not isinstance(self.model_DF["GAPRE"][interval_str], np.ndarray) or self.model_DF["GAPRE"][interval_str] is None:
            self.Entanglement_Entropy(interval, zero_modes=zero_modes)
       
        nu = np.real(self.model_DF["GAPRE"][interval_str])
        if debug:
            print(f'Eigs of Gamma_A_Partial:\n{nu}')
        nu_p = nu[(nu>=0)]
        nu_p[nu_p>=1] = 1-1e-14 #helps sort out issues
        nu_p = np.sort(nu_p)
        
    
        eps = - np.log(1/((1+nu_p)/2)-1)
        eps = np.sort(eps)
    
        ss_size = len(nu_p)

        number_of_eigenvalues = max_number_of_eigenvalues
        if max_number_of_eigenvalues > 2**ss_size:
            number_of_eigenvalues = 2*ss_size
        out = []
        if debug:
            print(f'Nus:\n{nu_p}')
            print(f'Nus:\n{eps}')

        nus = np.array([(1+nu_p)/2,(1-nu_p)/2])
        for i in range(number_of_eigenvalues):
            mask = (np.array([bool(int(n)) for n in np.binary_repr(i, width = ss_size)]))[::-1]
            out.append(np.sum(np.sum(eps[(mask)])))
        out = np.sort(np.array(out))
        out = out

        #for i in range(number_of_eigenvalues):
        #    mask = (np.array([bool(int(n)) for n in np.binary_repr(i, width = ss_size)]))[::-1]
        #    out.append(np.sum(nus[np.array([np.logical_not(mask), mask])]))
        #out = np.array(out)
        return out

        self.model_DF["ES"][interval_str] = out
        if Return:
            return self.model_DF["ES"][interval_str]

    def Modular_Hamiltonian_Log_Method(self, interval, sites = False,  zero_modes = 'filled', debug = False, return_zetas = False, return_spectrum = False, Redo =False, large_Sort_warning = 1000, SVD = False, hermitian = False, double_log_or_inverse = 'double_log'):
        if sites:
            site_list = interval
            interval = []
            for site in site_list:
                interval.append(site)
                interval.append(site)
        
        if self.Gamma_A[zero_modes] is None:
            self.Get_Gamma_A(Return = False, zero_modes=zero_modes, large_Sort_warning=large_Sort_warning)

        ss_sites = []
        for i in range(len(interval)//2):
            a_i = interval[2*i]
            b_i = interval[2*i+1]

            ss_sites += np.arange(a_i,b_i+1).tolist()

        if debug:
            print(f'ss_sites:\n{ss_sites}\n')
        
        L = len(ss_sites)

        Gamma_A_Partial = self.Get_Gamma_A_Partial(interval =  interval,  sites = sites,  zero_modes = zero_modes,  large_Sort_warning = large_Sort_warning)
        
        majorana_fermion_cor_mat = (1j*Gamma_A_Partial+np.eye(2*L))/2.
        majorana_fermion_cor_mat = majorana_fermion_cor_mat.astype(np.complex)
        
        if double_log_or_inverse == 'double_log':
            if SVD:
                res = -(Matrix_Log_SVD(np.eye(len(majorana_fermion_cor_mat))-majorana_fermion_cor_mat) - Matrix_Log_SVD(majorana_fermion_cor_mat))
            if not SVD:
                res = -(Matrix_Log(np.eye(len(majorana_fermion_cor_mat))-majorana_fermion_cor_mat, hermitian=hermitian) - Matrix_Log(majorana_fermion_cor_mat, hermitian=hermitian))

        if double_log_or_inverse == 'inverse':       
            if SVD:
                res = -(Matrix_Log_SVD(la.inv(majorana_fermion_cor_mat) - np.eye(len(majorana_fermion_cor_mat))))
            if not SVD:
                res = -Matrix_Log(la.inv(majorana_fermion_cor_mat) - np.eye(len(majorana_fermion_cor_mat)), hermitian=hermitian)
        
        return res

    def Modular_Hamiltonian_Fitting(self, interval, threshold = 10e-15, Jx = True, g = True, Jy = False, b_sigma = False, flip = [], subspace_intersection_search = 0, Extra_Couplings = [], Extra_Couplings_Symmetric = [], choose = 0, Return = 'Ham', Order = 'OpList', zero_modes = 'filled', \
                                                Symmetric = False, Redo = True, debug_round = 5, debug = False, Full_Debug = False, debug_With_Gamma_A_Partial = False, Full_Debug_Gamma_A_Partial = False, debug_G = False, \
                                                    Fit_Degenerate_Solutions = False, vec_for_fit = None, print_evals_P_ij = False, overlap_threshold = 10e-10, search_debug = False, print_levels = False, Allow_Two = False):
        sites = np.arange(interval[0], interval[1]+1)
        if debug: print(f'Sites: {sites}')
        L = len(sites)
        Gamma_A_Partial = self.Get_Gamma_A_Partial(interval = interval, zero_modes=zero_modes, Redo=Redo, debug = debug_With_Gamma_A_Partial, Full_Debug=Full_Debug_Gamma_A_Partial)
        schur = la.schur(Gamma_A_Partial)

        Gamma_C = schur[0]
        W = schur[1].T
        if debug and Full_Debug:
            print('Gamma_A_Partial:')
            PrintMatrix(Gamma_A_Partial, Round = debug_round)
            print('\n')
        if debug and Full_Debug:
            print('Gamma_C:')
            PrintMatrix(Gamma_C, Round = debug_round)
            print('\n')
        Gamma_C[Gamma_C<-10e-11] = -1
        Gamma_C[Gamma_C>10e-11] = 1
        if len(flip)>0 and not subspace_intersection_search:
            for i in flip:
                Gamma_C[2*i, 2*i+1] *= -1
                Gamma_C[2*i+1, 2*i] *= -1
        
        if debug and Full_Debug:
            print('Gamma Entanglement Ground State:')
            PrintMatrix(Gamma_C, Round = debug_round)
            print('\n')
        
        operators = []
        if not Symmetric:
            for i in range(L-1):
                site = sites[i]
                if Jx and self.Jx[site]!=0:
                    operators.append([(2*i+1)%(2*L), (2*i+2)%(2*L)])
                if Jy and self.Jy[site]!=0:
                    operators.append([(2*i)%(2*L), (2*i+3)%(2*L)])
                if b_sigma and not np.all(self.b_sigma==0):
                    if self.b_sigma[site] != 0:
                        operators.append([(2*i+1)%(2*L), (2*i+3)%(2*L)])

            if g:
                for i in range(L):
                    site = sites[i]
                    if self.g[site]!=0:
                        operators.append([(2*i)%(2*L), (2*i+1)%(2*L)])
        symmetric_operators = []
        if Symmetric:
            for i_L in range((L)//2): 
                site_L = sites[i_L]
                
                i_R = L - i_L-2
                site_R = L - i_R-1

                if Jx and self.Jx[site_L]!=0: 
                    op_L = [(2*i_L+1)%(2*L), (2*i_L+2)%(2*L)]
                    op_R = [(2*i_R+1)%(2*L), (2*i_R+2)%(2*L)]
                    symmetric_operators.append([op_L, op_R])

                if Jy and self.Jy[site_L]!=0:
                    op_L = [(2*i_L)%(2*L), (2*i_L+2)%(2*L)]
                    op_R = [(2*i_R)%(2*L), (2*i_R+2)%(2*L)]
                    symmetric_operators.append([op_L, op_R])

                #if b_sigma and not np.all(self.b_sigma==0):
                #    if self.b_sigma[i] != 0:
                #        operators.append([(2*i+1)%(2*L), (2*i+3)%(2*L)])

            if g:
                for i_L in range(L//2):
                    site_L = sites[i_L]
                    i_R = L - i_L - 1
                    site_R = sites[i_R]

                    if self.g[site_L]!=0:
                        op_L = [(2*i_L)%(2*L), (2*i_L+1)%(2*L)]
                        op_R = [(2*i_R)%(2*L), (2*i_R+1)%(2*L)]
                        symmetric_operators.append([op_L, op_R])
        


        if len(Extra_Couplings)!=0:
            for operator in Extra_Couplings:
                operators.append(operator)
        
        if len(Extra_Couplings_Symmetric)!= 0:
            for operator in Extra_Couplings_Symmetric:
                symmetric_operators.append(operator)

        if debug and Full_Debug: print(f'Operators: {operators}\n Symmetric Operators: {symmetric_operators}')

        operators_temp = operators
        operators = []
        [operators.append(x) for x in operators_temp if x not in operators]
        operators_temp = None

        symmetric_operators_temp = symmetric_operators
        symmetric_operators = []
        [symmetric_operators.append(x) for x in symmetric_operators_temp if x not in symmetric_operators]
        symmetric_operators_temp = None

        operators = operators + symmetric_operators
        
        if debug or Full_Debug: print(f'Number of Operators: {len(operators)}')
        if subspace_intersection_search == 0:
            restricted_propagator = W.T@(np.eye(len(Gamma_C)) + 1j*Gamma_C)@W
            if debug: print(f'Getting G')
            G = G_ab(operators, restricted_propagator, PrintMat = debug_G, debug=debug_G and Full_Debug)

            if debug and Full_Debug:
                if Full_Debug: 
                    print('\nG:')
                    PrintMatrix(G, Round = debug_round)
                    print('\n')
                eigvals, eigvecs = PrintEigensystem(G, Return = True, hermitian=True, Round = debug_round)
                print('\n')
            
            else:
                eigvals, eigvecs = la.eigh(G)

            idx = eigvals.argsort()
            eigvals = eigvals[idx]
            eigvecs = eigvecs[idx]

            if debug: print(f'Eigs of G: {eigvals}')
            
            if not Fit_Degenerate_Solutions:
                g0 = eigvals[choose]
                if debug: print(f'g0: {g0}\n')
                w = (eigvecs[:,choose].flatten())
                mod_ham = Get_Ham_From_Ops_And_Weights(operators, w, size = len(restricted_propagator), dtype=restricted_propagator.dtype)
            
            if Fit_Degenerate_Solutions:
                mask = (eigvals)<threshold
                eigvals_for_fit = eigvals[mask]
                eigvecs_for_fit = eigvecs[:,mask]
                
                if debug or Full_Debug: print(f'Fitting {len(eigvals_for_fit)} Degenerate Options')
                find_ham_fit = lambda a: la.norm(sum((a[2*i]+1j*a[2*i+1])*(Get_Ham_From_Ops_And_Weights(operators, eigvecs_for_fit[:,i], size = len(restricted_propagator), dtype=restricted_propagator.dtype)@vec_for_fit) - vec_for_fit for i in range(len(eigvals_for_fit)))) 
                initial_guess = np.ones(2*len(eigvals_for_fit))
                
                ham_fit_res = opt.minimize(find_ham_fit, initial_guess)
                if debug: print(f'\nResults for Fitting The Degenerate Hamiltonians:\n{ham_fit_res}\n')
            
                w = sum(eigvecs_for_fit[:,i]*(ham_fit_res.x[2*i]+1j*ham_fit_res.x[2*i+1]) for i in range(eigvecs_for_fit.shape[1]))
                mod_ham = Get_Ham_From_Ops_And_Weights(operators, w, size = len(restricted_propagator), dtype=restricted_propagator.dtype)

        at_one = False
        solutions_list = []
        if subspace_intersection_search > 0:
            Gamma_C_OG = Gamma_C.copy()
            
            P_dict = {}
            search_res_dict = {} 
            search_array = np.arange(subspace_intersection_search)
            if search_debug: print(f'Getting all the initial projectors!\n==================================================================================================================================\n')
            P_dict[0] = {}
            for i in search_array:            
                if debug or search_debug: print(f'Doing State {i}')
                to_flip = np.binary_repr(i, width = L)[::-1]
                if debug or Full_Debug: print(f'    Flipping Array: {to_flip}')
                Gamma_C = Gamma_C_OG.copy()
                for m in range(len(to_flip)):
                    if int(to_flip[m]) == 1:
                        Gamma_C[2*m, 2*m+1]*=-1
                        Gamma_C[2*m+1, 2*m]*=-1
                restricted_propagator = W.T@(np.eye(len(Gamma_C)) + 1j*Gamma_C)@W
                G = G_ab(operators, restricted_propagator, PrintMat = debug_G, debug=debug_G and Full_Debug)
                
                evals, evecs = la.eigh(G)
                idx = evals.argsort()
                evals = evals[idx]
                evecs = evecs[idx]
                
                mask = np.abs(evals) < threshold
                degenerate_vectors = evecs[:,mask]
                if debug or Full_Debug: print(f'    Number of Degenerate Vectors: {len(evals[mask])}\n')
                P = degenerate_vectors@np.conj(degenerate_vectors.T)
                P_dict[0][str([i])] = P.copy()
                if debug or search_debug: print(f'    Sum(abs(P^2-P)) = {np.linalg.norm(P@P-P)}')
                if debug or search_debug: print(f'----------------------------------')

            #IPd.clear_output()
            if debug or search_debug: print(f'\n\nBeginning Search!\n==================================================================================================================================\n')
            levels = np.arange(subspace_intersection_search-1)
            
            Found = False
            smallest_subspace = 2**L
            for level in levels:
                if debug or search_debug or print_levels: print(f'On level {level}. \n  Number Of Operators: {len(P_dict[level])}. Smallest Subspace from Previous Level: {smallest_subspace}')
                P_dict[level+1] = {}
                search_res_dict[level] = {}

                keys = list(P_dict[level].keys())
                for i in range(len(keys)-1):
                    starting_with_i_num_zeros = []
                    starting_with_i_dat = []
                    for j in np.arange(i+1, len(keys)):
                        key_i = keys[i]
                        key_j = keys[j]
                        key = str(ast.literal_eval(key_i)+ast.literal_eval(key_j))

                        if search_debug: print(f'   (key_i, key_j) = ({key_i}, {key_j}). key = {key}')

                        P_i = P_dict[level][key_i]
                        P_j = P_dict[level][key_j]

                        P_ij = P_i@P_j

                        #evecsP_ij, evalsP_ij, _ = la.svd(P_ij)
                        evalsP_ij, evecsP_ij = la.eig(P_ij)
                        mask = np.abs(evalsP_ij - 1)< overlap_threshold
                        
                        if len(evalsP_ij[mask])<smallest_subspace:
                            smallest_subspace = len(evalsP_ij[mask])
                        if search_debug: print(f'       Num of 1s: {len(evalsP_ij[mask])}')
                        if search_debug and print_evals_P_ij: print(f'      evals_P_ij:\n{evalsP_ij}\n------------------------------------------------------------------------------------\n')
                        if len(evalsP_ij[mask]) == 1:
                            Found = True
                            #solutions_list.append(key)
                            break

                        if len(evalsP_ij[mask]) == 2 and Allow_Two:
                            Found = True
                            if debug or Full_Debug: print(f'Found Two Possible Solutions!')
                            break
                        
                        
                        starting_with_i_num_zeros.append(len(evalsP_ij[mask]))
                        starting_with_i_dat.append([key, P_ij.copy()])

                    if Found:
                        break
                    starting_with_i_num_zeros = np.array(starting_with_i_num_zeros)
                    starting_with_i_dat = np.array(starting_with_i_dat, dtype=object)

                    mask = (starting_with_i_num_zeros == starting_with_i_num_zeros.min())
                    for dat in starting_with_i_dat[mask]:
                        P_dict[level+1][dat[0]] = dat[1].copy()
                    
                    #P = np.eye(len(starting_with_i_dat[0][1]))
                    #KEY = []
                    #for dat in starting_with_i_dat[mask]:
                    #    P = P@dat[1]
                    #    KEY = KEY + ast.literal_eval(dat[0])
                    
                    #KEY0 = []
                    #[KEY0.append(x) for x in KEY if x not in KEY0]
                    #KEY0 = str(KEY0)
                    #P_dict[level+1][KEY0] = P.copy()
                    

                    if search_debug: print(f'  For {i}, keeping {len(starting_with_i_num_zeros[mask])} Projectors')        

                    
                #if not Found:
                    #IPd.clear_output()                        
                if Found:
                    break
            

            if Fit_Degenerate_Solutions:
                evals_to_fit = evalsP_ij[mask]
                evecs_to_fit = evecsP_ij[:,mask]
                con = lambda a: 1-np.sqrt(sum((a[2*i]+1j*a[2*i+1])*a[2*i]-1j*a[2*i+1] for i in range(len(evals_to_fit))))
                cons = {'type': 'eq', 'fun':con}
                #find_ham_fit = lambda a: la.norm(sum((a[2*i]+a[2*i+1]*1j)*(Get_Ham_From_Ops_And_Weights(operators, evecs_to_fit[:,i], size = len(restricted_propagator), dtype=restricted_propagator.dtype)@vec_for_fit) - vec_for_fit for i in range(len(evals_to_fit)))) 
                find_ham_fit = lambda a: np.abs(np.imag(sum((a[2*i]+1j*a[2*i+1])*evecs_to_fit[:,i] for i in range(len(evals_to_fit))))).sum()

                initial_guess = np.ones(2*len(evals_to_fit))
                ham_fit_res = opt.minimize(find_ham_fit, initial_guess, constraints = cons)
                if debug: print(f'\nResults for Fitting The Degenerate Hamiltonians:\n{ham_fit_res}\n')
            
                w = sum(evecs_to_fit[:,i]*(ham_fit_res.x[2*i]+1j*ham_fit_res.x[2*i+1]) for i in range(len(initial_guess)//2))

                mod_ham = Get_Ham_From_Ops_And_Weights(operators, w, size = len(restricted_propagator), dtype=restricted_propagator.dtype)
            
            else:        
                idx = evalsP_ij.argsort()[::-1]
                evalsP_ij = evalsP_ij[idx]
                evecsP_ij = evecsP_ij[idx]
                mask = np.abs(1-evalsP_ij)<overlap_threshold
                if len(evalsP_ij[mask])==1:
                    w = evecsP_ij[:, 0].flatten()
                if len(evalsP_ij[mask])>1:
                    w = evecsP_ij[:, choose].flatten()
                mod_ham = Get_Ham_From_Ops_And_Weights(operators, w, size = len(restricted_propagator), dtype=restricted_propagator.dtype)

        if debug and Full_Debug:
            print('Initial Mod Ham:')
            PrintMatrix(mod_ham, Round = 3)
            print('\n')        

        entropy = self.Entanglement_Entropy(interval, zero_modes=zero_modes, Redo=Redo)
        if debug: print(f'\nTarget Entropy: {entropy}')
        entropy_compare = lambda a: np.abs(entropy - Entropy_From_H(1j*(a*mod_ham)))
        
        res = opt.minimize(entropy_compare, [1])

        mod_ham = res.x*mod_ham
        if debug: print(f'{res}\n')
        if debug: print(f'Resulting Entropy: {Entropy_From_H(1j*mod_ham)}')

        if mod_ham[0,1]<0:
            mod_ham *= -1

        if Return == 'Ham':
            return mod_ham
        if Return == 'Weights':
            if Order == 'Bonds':
                weights = []
                for i in range(len(mod_ham)-1):
                    weights.append(mod_ham[i,i+1])
                return weights
            if Order == 'OpList':
                return w*res.x

        if Return == 'Both': 
            if Order == 'Bonds':
                weights = []
                for i in range(len(mod_ham)-1):
                    weights.append(mod_ham[i,i+1])
                return [mod_ham, weights]
            return [mod_ham, w.res.x]

    def Modular_Hamiltonian_From_Beta(self, interval, debug = False, F = "p_chain", half_steps = True, const = 0):
        '''Current string options for F are: p_chain (periodic chain, no defect) and open_on_boundary for open chain with interval on the boundary.'''
        if self.Ham is None:
            self.Ham_Builder()
        
        
        sites = np.linspace(interval[0], interval[1], (interval[1]-interval[0])+1, dtype = int)
        if debug:
            print(f'sites:\n{sites}')
        L = len(sites)
        if debug:
            print(f'Number of sites = {L}')
        
        picked_F = False
        if type(F) != str:
            picked_F = True
        
        if not picked_F:
            if F == "p_chain":
                F = lambda x: Periodic_Beta(x+0.5, L, self.N) #lambda x: np.sin(np.pi*(x+0.5)/self.N)*np.sin(np.pi*(L-x-0.5)/self.N) / (np.sin(np.pi*L/self.N))*4*self.N if L!=self.N else 0
            
            if F == "open_on_boundary":
                F = lambda x: Open_On_Boundary_Beta(x+0.5, L, self.N) #lambda x: 4*self.N*(np.cos(np.pi*(x+0.5)/(self.N)) - np.cos(np.pi*(L)/(self.N)))/(2*np.sin(np.pi*(L)/(self.N)))



        Modular_Ham = np.zeros((2*L,2*L),dtype=complex)
        
        
        if debug:
            print("Doing g Bits")
        
        X = np.arange(L)
        
        X = np.arange(0,L)

        for i in range(len(sites)):
            site = sites[i]
            if debug:
                print(f'    i: {i}, site={site}')
            x = X[i]
            #Modular_Ham[2*i:2*(i+1),2*i:2*(i+1)] = F(x)*self.Ham[2*site:2*(site+1),2*site:2*(site+1)]
            m = 2*i
            n = 2*i+1
            if debug:
                print(f'        F(x) = {round(F(x))}')
            Modular_Ham[m,n] = F(x)*self.Ham[m,n]
            Modular_Ham[n,m] = F(x)*self.Ham[n,m]
            

        if debug:
            print('Doing Jx, Jy Bits')

        

        X = np.arange(0,L-1)+0.5
        for i in  range(len(sites)-1):
            site = sites[i]
            if debug:
                print(f'    i: {i}, site={site}')
            
            x = X[i]
            Modular_Ham[2*i+2,2*i+1] = F(x)*self.Ham[2*site+2, 2*site+1]
            Modular_Ham[2*i+1,2*i+2] = F(x)*self.Ham[2*site+1, 2*site+2]
            
            Modular_Ham[2*i,2*i+3] = F(x)*self.Ham[2*site, 2*site+3]
            Modular_Ham[2*i+3,2*i] = F(x)*self.Ham[2*site+3, 2*site]

        return 2*Modular_Ham+const*np.eye(len(Modular_Ham))

    def Entanglement_Spectrum_From_Beta(self, interval, debug = False, max_number_of_eigenvalues = 100, F = None):
        if F is None:
            Modular_Ham = self.Modular_Hamiltonian_From_Beta(interval, debug = debug)
        else:
            Modular_Ham = self.Modular_Hamiltonian_From_Beta(interval, debug = debug, F = F)
        if debug:
            print('Modular Ham Acquired')
        Eigs = la.eigh(Modular_Ham)[0]
        if debug:
            print(f"Eigs:\n{Eigs}")
        
        
        nu = np.real(Eigs)
        nu_p = nu[(nu>10**-self.cutoff)]
        #nu_p[nu_p>=1] = 1 #helps sort out issues
        #nu_p = np.sort(nu_p)

        if debug:
            print(f'nu_p:\n{nu_p}')

        nus = np.array([nu_p,nu_p])#np.array([(1+nu_p)/2,(1-nu_p)/2])
        ss_size = len(nu_p)

        number_of_eigenvalues = max_number_of_eigenvalues
        if max_number_of_eigenvalues > 2**ss_size:
            number_of_eigenvalues = 2*ss_size
        out = []
        if debug:
            print(f'Nus:\n{nus}')
        for i in range(number_of_eigenvalues):
            mask = (np.array([bool(int(n)) for n in np.binary_repr(i, width = ss_size)]))[::-1]
            out.append(np.product(nus[np.array([np.logical_not(mask), mask])]))
        out = np.array(out)
        return out

    def Fermionic_Logarithmic_Negativity(self, interval, P = 2, Return = True, debug = False, print_mask = False, sites = False, Redo = False, zero_modes = 'filled'):
        '''P indexes with first interval being 1. This is done using the r sector \tilde{Gamma} from 1906.04211. Specifically, this calculates equation 39. This is required as \tilde{Gamma} is hermitian.'''
        if type(P) is int:
            P = [P]
        
        if sites:
            site_list = interval
            interval = []
            for site in site_list:
                interval.append(site)
                interval.append(site)

    
        # This is the same code as entanglement negativity
        if debug:
            print('This is the same stuff as in entanglement entropy')

        interval_str = str(interval)+str("_P=")+str(P)
        interval = np.array(interval).astype(int)
        xi_arr = interval[::2]%(self.N)
        xf_arr = interval[1::2]%(self.N)
    
        
        if len(xi_arr) != len(xf_arr):
            print("Bad Interval")
            return None

        if interval_str not in self.model_DF.index:
            self.model_DF.loc[interval_str] = np.empty(len(self.model_DF.columns), dtype=object)
        
        if not self.model_DF["FLN"][interval_str] == None and not Redo:
            if Return: 
                return self.model_DF["FLN"][interval_str] 
            else:
                return None
        
        
        if self.Gamma_A[zero_modes] is None:
            self.Get_Gamma_A(Return = False, zero_modes=zero_modes)

        mask = np.zeros((2*self.N,2*self.N), dtype=bool)
        false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
        for i in range(len(xi_arr)):
            xi = xi_arr[i]
            xf = xf_arr[i]
            index_arr = np.indices((2*self.N,2*self.N))
            tf_start =  index_arr >= (2*xi)
            if xi >= 0:
                tf_start = ~tf_start
            tf_end = index_arr < 2*(xf+1)
            or_xy = np.logical_xor(tf_start,tf_end)
            mask_i = np.logical_and(or_xy[0],or_xy[1])
            if debug:
                print(f"Mask for interval {i+1}")
                PrintMatrix(np.int8(mask_i))
            mask = np.logical_or(mask, mask_i)
        
        false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
            
        if debug or print_mask:
            print("Mask:")
            PrintMatrix(np.int8(mask))

        
        Gamma_A_Partial = np.delete(np.delete(self.Gamma_A[zero_modes],false_loc,axis=0), false_loc, axis = 1)
        if debug:
            PrintMatrix(Gamma_A_Partial, Round = 4)
        Gamma_A_Partial = -1j*Gamma_A_Partial.astype("complex64")
        
    
        # This is where it differs form the entanglement entropy
        if debug:
            print('This is the stuff for as in log negativity')
        logZ_PT = 0 # i initially had this initialize as 1. idk why
        U = np.identity(len(Gamma_A_Partial), dtype = int)
        identity = np.identity(len(U), dtype = int)
        
        if debug:
            test = np.zeros((len(U),len(U)), dtype=int)
        
        if debug:
            print('Starting Gamma_A_Partial Eigenvalues (Should come in doubles)\n')
            print(np.round(la.eigvals(Gamma_A_Partial),4))
        l_arr = []
        for i in range(len(xf_arr)):
            l_arr.append(xf_arr[i]-xi_arr[i]+1)
    
        l_arr = np.array(l_arr,dtype = int)
        for i in P:
            a = 2*np.sum(l_arr[:i-1])
            b = a + 2*l_arr[i-1]
            if debug:
                print(f'For interval {i}, a = {a} and b = {b}')
            Gamma_A_Partial[b:, a:b] *= -1.j # selects below
            Gamma_A_Partial[a:b, b:] *= -1.j # selects right
            Gamma_A_Partial[:a, a:b] *= -1.j # selects above
            Gamma_A_Partial[a:b, :a] *= -1.j # selects left
            Gamma_A_Partial[a:b, a:b] *= -1. # selects interval
            #return(Gamma_A_Partial[a:b, a:b])
            U[a:b,a:b] *= -1
            if debug:
                print(f'|Z_PT| = {np.abs(np.sqrt(np.abs(np.linalg.det(Gamma_A_Partial[a:b, a:b]))))}')
            logZ_PT += np.log(np.abs(np.sqrt(np.abs(np.linalg.det(Gamma_A_Partial[a:b, a:b])))))

            if debug:
                test[a:b,b:] += 2 # selects right
                test[b:,a:b] += 2 # selects below
                test[:a, a:b] += 2 # selects above
                test[a:b, :a] += 2 # selects left
                test[a:b, a:b] += -1 # selects interval
        
        if debug:
            print("Matrix of changes for partial transpose:")
            PrintMatrix(test)
            print("\n U Matrix:")
            PrintMatrix(U)
            print("\n Gamma A Partial Post Partial Transpose")
            PrintMatrix(Gamma_A_Partial,4)
        
                
        if debug:
            print("Shape of Gamma_A: "+str(self.Gamma_A[zero_modes].shape))
            print("Shape of Gamma_A_Partial: "+str(Gamma_A_Partial.shape))
            print("DataType = "+str(type(Gamma_A_Partial[0][0])))
            
        if Gamma_A_Partial.size == 0:
            self.model_DF["FLN"][interval_str] = 0
            self.model_DF["GAPTE"][interval_str] = np.array([0])
        
        else: 
            Exp_Omega_Tilde = ((identity+Gamma_A_Partial)@la.inv(identity-Gamma_A_Partial))@U

            if debug:
                print(f'\nExp_Omega_Tilde:')
                PrintMatrix(Exp_Omega_Tilde)
                #PrintMatrix((identity+Gamma_A_Partial)@la.inv(identity-Gamma_A_Partial))
                print(f"Checking Correct inverse of Exp_Omega_Tilde (should be Identity)")
                PrintMatrix((identity-Exp_Omega_Tilde)@la.inv(identity-Exp_Omega_Tilde),Round = 2, buffer = 2)
                print(f"Checking conj(Gamma_A_Partial.T) - U.Gamma_A_Partial.U = 0:     <- Gamma_A_Partial is pseudo-Hermitian")
                PrintMatrix(np.conj(Gamma_A_Partial.T) - U@Gamma_A_Partial@U, Round = 2,buffer = 2)
                hmmmReal = np.real(Exp_Omega_Tilde - np.conj(Exp_Omega_Tilde.T))
                hmmmImag = np.imag(Exp_Omega_Tilde - np.conj(Exp_Omega_Tilde.T))
                print(f"Checking if (1+Exp_Omega_Tilde).(1-Exp_Omega_Tilde)^-1.U is Hermitian (should be zero):")
                PrintMatrix(hmmmReal + 1j*hmmmImag, Round = 2, buffer = 2)

            Eigs = np.round(la.eig(Exp_Omega_Tilde)[0], self.cutoff)
        
            if debug:
                print(f'Eigs of Exp_Omega_Tilde:\n{np.round(Eigs,4)}')
            Gamma_Tilde_Eigs = np.round(np.tanh(np.log(Eigs)/2.), self.cutoff)
            self.model_DF["GAPTE"][interval_str] = Eigs
            
            if debug:
                print("Eigs of Exp_Gamma_Tilde: \n"+str(Gamma_Tilde_Eigs))

            nu_arr = Gamma_Tilde_Eigs[np.real(Gamma_Tilde_Eigs)>10**-self.cutoff]
            if debug:
                print("Nus: "+str(nu_arr))
                print(f"2*len(nu_arr)-len(Eigs): {2*len(nu_arr) - len(Eigs)}.")
            
            Sl_arr = np.log(np.abs((1+nu_arr)/2.)+np.abs((1-nu_arr)/2.))
            
            for i in range(len(Sl_arr)):
                if np.isnan(Sl_arr[i]):
                    Sl_arr[i] = 0
                
            
            self.model_DF["FLN"][interval_str] = np.round(np.sum(Sl_arr)+logZ_PT,self.cutoff)
        
        if Return:
            return self.model_DF["FLN"][interval_str]

    def Fermionic_Logarithmic_Negativity_NS(self, interval, P = 2, Return = True, debug = False, print_mask = False, sites = False, Redo = False, zero_modes = 'filled'):
        '''P indexes with first interval being 1. This is done using the ns sector Gamma from 1906.04211. Specifically, this calculates equation 36 for n = 1/2.'''
        if type(P) is int:
            P = [P]
        
        if sites:
            site_list = interval
            interval = []
            for site in site_list:
                interval.append(site)
                interval.append(site)

    
        # This is the same code as entanglement negativity
        if debug:
            print('This is the same stuff as in entanglement entropy')

        interval_str = str(interval)+str("_P=")+str(P)
        interval = np.array(interval).astype(int)
        xi_arr = interval[::2]%(self.N)
        xf_arr = interval[1::2]%(self.N)
    
        
        if len(xi_arr) != len(xf_arr):
            print("Bad Interval")
            return None

        if interval_str not in self.model_DF.index:
            self.model_DF.loc[interval_str] = np.empty(len(self.model_DF.columns), dtype=object)
        
        if not self.model_DF["FLN"][interval_str] == None and not Redo:
            if Return: 
                return self.model_DF["FLN"][interval_str] 
            else:
                return None
        
        
        if self.Gamma_A[zero_modes] is None:
            self.Get_Gamma_A(Return = False, zero_modes=zero_modes)

        mask = np.zeros((2*self.N,2*self.N), dtype=bool)
        false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
        for i in range(len(xi_arr)):
            xi = xi_arr[i]
            xf = xf_arr[i]
            index_arr = np.indices((2*self.N,2*self.N))
            tf_start =  index_arr >= (2*xi)
            if xi >= 0:
                tf_start = ~tf_start
            tf_end = index_arr < 2*(xf+1)
            or_xy = np.logical_xor(tf_start,tf_end)
            mask_i = np.logical_and(or_xy[0],or_xy[1])
            if debug:
                print(f"Mask for interval {i+1}")
                PrintMatrix(np.int8(mask_i))
            mask = np.logical_or(mask, mask_i)
        
        false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
            
        if debug or print_mask:
            print("Mask:")
            PrintMatrix(np.int8(mask))

        
        Gamma_A_Partial = np.delete(np.delete(self.Gamma_A[zero_modes],false_loc,axis=0), false_loc, axis = 1)
        if debug:
            PrintMatrix(Gamma_A_Partial, Round = 4)
        Gamma_A_Partial = -1j*Gamma_A_Partial.astype("complex64")
        
    
        # This is where it differs form the entanglement entropy
        if debug:
            print('This is the stuff for as in log negativity')
        logZ_PT = 1
        U = np.identity(len(Gamma_A_Partial), dtype = int)
        identity = np.identity(len(U), dtype = int)
        
        if debug:
            test = np.zeros((len(U),len(U)), dtype=int)
        
        if debug:
            print('Starting Gamma_A_Partial Eigenvalues (Should come in doubles)\n')
            print(np.round(la.eigvals(Gamma_A_Partial),4))
        l_arr = []
        for i in range(len(xf_arr)):
            l_arr.append(xf_arr[i]-xi_arr[i]+1)
    
        l_arr = np.array(l_arr,dtype = int)
        for i in P:
            a = 2*np.sum(l_arr[:i-1])
            b = a + 2*l_arr[i-1]
            if debug:
                print(f'For interval {i}, a = {a} and b = {b}')
            Gamma_A_Partial[b:, a:b] *= -1.j # selects below
            Gamma_A_Partial[a:b, b:] *= -1.j # selects right
            Gamma_A_Partial[:a, a:b] *= -1.j # selects above
            Gamma_A_Partial[a:b, :a] *= -1.j # selects left
            Gamma_A_Partial[a:b, a:b] *= -1. # selects interval
            #return(Gamma_A_Partial[a:b, a:b])
            U[a:b,a:b] *= -1
            logZ_PT += np.log(np.abs(np.sqrt(np.abs(np.linalg.det(Gamma_A_Partial[a:b, a:b])))))

            if debug:
                test[a:b,b:] += 2 # selects right
                test[b:,a:b] += 2 # selects below
                test[:a, a:b] += 2 # selects above
                test[a:b, :a] += 2 # selects left
                test[a:b, a:b] += -1 # selects interval
        
        if debug:
            print("Matrix of changes for partial transpose:")
            PrintMatrix(test)
            print("\n U Matrix:")
            PrintMatrix(U)
            print("\n Gamma A Partial Post Partial Transpose")
            PrintMatrix(Gamma_A_Partial,4)
        
                
        if debug:
            print("Shape of Gamma_A: "+str(self.Gamma_A[zero_modes].shape))
            print("Shape of Gamma_A_Partial: "+str(Gamma_A_Partial.shape))
            print("DataType = "+str(type(Gamma_A_Partial[0][0])))
            
        if Gamma_A_Partial.size == 0:
            return 0
        
        else: 
            Eigs = np.round(la.eigvals(Gamma_A_Partial),self.cutoff)
            if debug:
                print("Eigenvalues: \n"+str(Eigs))
                print("Gamma Eigenvalues: \n"+str(Eigs)) 
            
            nu_arr = Gamma_A_Partial[np.real(Gamma_A_Partial)>10**-self.cutoff]
            if debug:
                print("Nus: "+str(nu_arr))
                print(f"2*len(nu_arr)-len(Eigs): {2*len(nu_arr) - len(Eigs)}.")
            
            Sl_arr = np.log(np.abs((1+nu_arr)/2.)+np.abs((1-nu_arr)/2.))
            
            for i in range(len(Sl_arr)):
                if np.isnan(Sl_arr[i]):
                    Sl_arr[i] = 0
                
            
            self.model_DF["FLN"][interval_str] = np.round(np.sum(Sl_arr))
        
        if Return:
            return self.model_DF["FLN"][interval_str]

    def Get_Entanglement_Negativities(self, intervals, P = 2, Return = False, Print = False, sites = False):
        results = []
        for interval in intervals:
            if Print:
                print("Doing Interval = "+str(interval)+"\n")
            results.append(self.Fermionic_Logarithmic_Negativity(interval, P = P, Return=True, sites = sites))
            if Print:
                IPd.clear_output()
        if Return:
            return np.array(results)

    def Entropy_with_Spectrum(self, interval, max_number_of_eignvalues = 100, Round = 10, sites = False):
        eigs = self.Entanglement_Spectrum(interval, max_number_of_eigenvalues=max_number_of_eignvalues, Return=True, sites = sites)
        
        eigs = eigs[eigs>0]
        return np.round(-np.sum(eigs*np.log(eigs)), Round)

    def Plot_Entanglement_Entropy(self, interval_list, x_axis = None, title = 0, debug = False, size = None, dpi = None, return_points = False, sites = False, connected = False, zero_modes = 'filled'):
        entropy_list = []
        for interval in interval_list:
            entropy_list.append(self.Entanglement_Entropy(interval, sites = sites, zero_modes='filled'))
        if x_axis is None:
            x_axis = np.arange(len(interval_list))
        
        plt.figure(dpi = dpi, figsize=size)
        #print(x_axis)
        #print(entropy_list)

        plt.plot(x_axis, entropy_list, marker="D")

        plt.show()

        if return_points:
            return [x_axis, entropy_list]
    
    def GroundStateExpectationValues(self, operator, operator_name = None, debug = False):
        if self.Ground_State is None:
            self.Get_Ground_State_Energy(Return = True)
        
        if type(operator_name) == str:
            if operator_name in self.ExpectationValues.keys:
                return self.ExpectationValues[operator_name]
        
        res = np.conj(self.Ground_State.T) @ operator @ self.Ground_State

        if type(operator_name) == str:
            self.ExpectationValues[operator_name] = res
        
        return res
    
    def Spin_2_Site_RDM(self, sites, debug = False, print_mask = False, Redo = True, zero_modes = 'filled', PT2 = False):
        
        ''' Note that this entanglement entropy code does not do the empty interval, the interval [a, a] is treated as just the 'a' site. This is why [0,0] returns non-zero, it is the single site entropy.'''

        if len(sites) != 2:
            print("Wrong two sites, try again.")
            return None
        
        
        delta = lambda j,k: 1 if j==k  else 0
        
        if self.Gamma_A[zero_modes] is None:
            self.Get_Gamma_A(zero_modes = zero_modes, Return = False)
        s1 = sites[0] % self.N    
        s2 = sites[1] % self.N

        
        if debug:
            print(f'[s1,s2] = {[s1,s2]}')
        
        L = s2-s1
        if L==0:
            return 0

        majorana_cor = lambda m,n: delta(m,n) + 1j*self.Gamma_A[zero_modes][m,n]

        complex_fermion_cor = lambda j,k: (majorana_cor(2*j, 2*k)- 1j*majorana_cor(2*j, 2*k+1) + 1j*majorana_cor(2*j+1, 2*k) + majorana_cor(2*j+1, 2*k+1))/4

        # sigma_xx -------------------------------------------------------
        if debug:
            print(f'Building sigma_xx')
        sigma_xx_mat = np.zeros((L, L), dtype = complex)
        for index_j in range(L):
            for index_k in range(L):
                j = index_j+s1
                k = index_k+s1+1
                m = 2*j+1
                n = 2*k
                if debug:
                    print(f'(index_j,index_k) = ({index_j},{index_k})')
                    print(f'(j,k) = ({j},{k})')
                    print(f'(m,n) = ({m},{n})')
                    print(f'    <a_m a_n> = {majorana_cor(m,n)}\n')
                sigma_xx_mat[index_j, index_k] = majorana_cor(m,n) #1j*majorana_cor(m,n) #complex_fermion_cor(j,k)#
        
        if debug:
            print("sigma_xx:")
            PrintMatrix(sigma_xx_mat, Round = 4)   
            print(f"\nDet of sigma_xx_mat: {la.det(sigma_xx_mat)}\n")

        sigma_xx = (1j)**(L)*la.det(sigma_xx_mat)#*(1j)**(s2-s1)

        # sigma_xy -------------------------------------------------------
        if debug:
            print(f'Building sigma_xy')
        sigma_xy_mat = np.zeros((L, L), dtype = complex)
        for index_j in range(L):
            for index_k in range(L):
                j = index_j+s1
                k = index_k+s1+1
                m = 2*j+1
                n = 2*k
                if index_k == L - 1:
                    n = 2*k+1
                if debug:
                    print(f'(index_j,index_k) = ({index_j},{index_k})')
                    print(f'(j,k) = ({j},{k})')
                    print(f'(m,n) = ({m},{n})')
                    print(f'    <a_m a_n> = {majorana_cor(m,n)}\n')
                sigma_xy_mat[index_j, index_k] = majorana_cor(m,n) #1j*majorana_cor(m,n) #complex_fermion_cor(j,k)#
                #if index_j == L - 1:
                #    sigma_xx_mat[index_j, index_k] *= -1/1j
        
        if debug:
            print("sigma_xy:")
            PrintMatrix(sigma_xx_mat, Round = 4)   
            print(f"\nDet of sigma_xy_mat: {la.det(sigma_xy_mat)}\n")

        sigma_xy = -(1j)**(L)*la.det(sigma_xy_mat)
        

        # sigma_yx -------------------------------------------------------
        if debug:
            print(f'Building sigma_yx')
        sigma_yx_mat = np.zeros((L, L), dtype = complex)
        for index_j in range(L):
            for index_k in range(L):
                j = index_j+s1
                k = index_k+s1+1
                m = 2*j+1
                n = 2*k
                if index_j == 0:
                    m = 2*j
                if debug:
                    print(f'(index_j,index_k) = ({index_j},{index_k})')
                    print(f'(j,k) = ({j},{k})')
                    print(f'(m,n) = ({m},{n})')
                    print(f'    <a_m a_n> = {majorana_cor(m,n)}\n')
                sigma_yx_mat[index_j, index_k] = majorana_cor(m,n) #1j*majorana_cor(m,n) #complex_fermion_cor(j,k)#
                #if index_j == 0:
                #    sigma_xx_mat[index_j, index_k] *= 1/1j
        
        if debug:
            print("sigma_yx:")
            PrintMatrix(sigma_xx_mat, Round = 4)   
            print(f"\nDet of sigma_yx_mat: {la.det(sigma_yx_mat)}\n")

        sigma_yx = (1j)**(L)*la.det(sigma_yx_mat)


        # sigma_yy -------------------------------------------------------
        if debug:
            print(f'Building sigma_yy')
        sigma_yy_mat = np.zeros((s2-s1,s2-s1), dtype = complex)
        for index_j in range(s2-s1):
            for index_k in range(s2-s1):
                j = index_j+s1+1
                k = index_k+s1
                m = 2*j+1
                n = 2*k
                if debug:
                    print(f'(index_j,index_k) = ({index_j},{index_k})')
                    print(f'(j,k) = ({j},{k})')
                    print(f'(m,n) = ({m},{n})')
                    print(f'    <a_m a_n> = {majorana_cor(m,n)}\n')
                sigma_yy_mat[index_j, index_k] = majorana_cor(m,n) #1j*majorana_cor(m,n) #complex_fermion_cor(j,k)#
        
        sigma_yy = (1j)**(L)*la.det(sigma_yy_mat)#*(1j)**(s2-s1)

        if debug:
            print("sigma_yy:")
            PrintMatrix(sigma_yy_mat, Round = 4)   
            print(f"\nDet of sigma_yy_matl: {la.det(sigma_yy_mat)}")


        if debug:
            print(f'majoran_cor({2*s1}, {2*s1+1}) = {majorana_cor(2*s1, 2*s1+1)}')
            print(f'majoran_cor({2*s2}, {2*s2+1}) = {majorana_cor(2*s2, 2*s2+1)}')
        sigma_z = lambda j: -1*(2*complex_fermion_cor(j,j) - 1)
        #sigma_z_12 = lambda j,k: -4*complex_fermion_cor(j,k)*complex_fermion_cor(k,j)
        sigma_zz_mat = np.zeros((2,2), dtype = complex)
        index_list = [s1, s2]
        for index_j in range(2):
            for index_k in range(2):
                j = index_list[index_j]
                k = index_list[index_k]
                m = 2*j+1
                n = 2*k
                sigma_zz_mat[index_j, index_k] = 1j*majorana_cor(m,n)
        sigma_zz = la.det(sigma_zz_mat)
        if debug:
            print(f'sigma_z({s1}) = {sigma_z(s1)}')
            print(f'sigma_z({s2}) = {sigma_z(s2)}')
            print(f'sigma_z_12({s1},{s2}) = {sigma_zz}')
            print(f'sigma_xx({s1}, {s2}) = {sigma_xx}')
            print(f'sigma_xy({s1}, {s2}) = {sigma_xy}')
            print(f'sigma_yx({s1}, {s2}) = {sigma_yx}')
            print(f'sigma_yy({s1}, {s2}) = {sigma_yy}')
        a11 = 1/4*(1+sigma_z(s1)+sigma_z(s2)+sigma_zz)
        a22 = 1/4*(1+sigma_z(s1)-sigma_z(s2)-sigma_zz)
        a33 = 1/4*(1-sigma_z(s1)+sigma_z(s2)-sigma_zz)
        a44 = 1/4*(1-sigma_z(s1)-sigma_z(s2)+sigma_zz)
        a14 = (sigma_xx + 1j*sigma_yx + 1j* sigma_xy - sigma_yy)/4
        a23 = (sigma_xx + 1j*sigma_yx - 1j* sigma_xy + sigma_yy)/4 
        a32 = (sigma_xx - 1j*sigma_yx + 1j* sigma_xy + sigma_yy)/4 
        a41 = (sigma_xx - 1j*sigma_yx - 1j* sigma_xy - sigma_yy)/4 
        
        if PT2:
            a14 = (sigma_xx - 1j*sigma_yx + 1j* sigma_xy + sigma_yy)/4
            a23 = (sigma_xx - 1j*sigma_yx - 1j* sigma_xy - sigma_yy)/4 
            a32 = (sigma_xx + 1j*sigma_yx + 1j* sigma_xy - sigma_yy)/4 
            a41 = (sigma_xx + 1j*sigma_yx - 1j* sigma_xy + sigma_yy)/4 

        return np.round(np.array([[a11, 0,   0,   a41],\
                                  [0,   a22, a32, 0],\
                                  [0,   a23, a33, 0],\
                                  [a14, 0,   0,   a44]]), self.cutoff)

    def Spin_2_Site_Entropy(self, sites, debug = False, print_mask = False, Redo = True, zero_modes = 'filled', mi = False):
        '''mi=True calculates mutual information as well'''
        rdm = self.Spin_2_Site_RDM(sites, debug=debug, print_mask=print_mask, Redo = Redo, zero_modes=zero_modes)
        
        if debug:
            print("RDM:")
            PrintMatrix(rdm, Round = 4)
        w = la.eigh(rdm)[0]
        if debug:
            print(f"Eigenvalues of the RDM:\n{w}")

        S = 0
        for eig in w:
            if debug:
                    print(f'current eig is {eig}')
            if np.abs(eig) > 10**(-1*self.cutoff):
                if debug:
                    print(f'|Eig| is bigger than zero')
                S+= -1*eig*np.log(eig)

        if mi:
            v1 = np.array([1,0])
            v2 = np.array([0,1])
            mat1 = np.kron(v1, np.identity(2))
            mat2 = np.kron(v2, np.identity(2))
            rdmA = mat1@rdm@mat1.T + mat2@rdm@mat2.T
            if debug:
                print(f'rdmA:')
                PrintMatrix(rdmA)

            w = la.eigh(rdmA)[0]
            if debug:
                print(f"Eigenvalues of the rdmA:\n{w}")

            SA = 0
            for eig in w:
                if debug:
                        print(f'current eig is {eig}')
                if np.abs(eig) > 10**(-1*self.cutoff):
                    if debug:
                        print(f'|Eig| is bigger than zero')
                    SA+= -1*eig*np.log(eig)

            mat1 = np.kron(np.identity(2), v1)
            mat2 = np.kron(np.identity(2), v2)
            rdmB = mat1@rdm@mat1.T + mat2@rdm@mat2.T
            if debug:
                print(f'rdmB:')
                PrintMatrix(rdmB)

            SB = 0
            for eig in w:
                if debug:
                        print(f'current eig is {eig}')
                if np.abs(eig) > 10**(-1*self.cutoff):
                    if debug:
                        print(f'|Eig| is bigger than zero')
                    SB+= -1*eig*np.log(eig)

            return [S, SA, SB, SA+SB-S]

        return S
    
    def Spin_2_Site_Log_Negativity(self, sites, debug = False, print_mask = False, Redo = True, zero_modes = 'filled'):
        rdm = self.Spin_2_Site_RDM(sites, debug=debug, print_mask=print_mask, Redo = Redo, zero_modes=zero_modes, PT2 = True)
        
        if debug:
            print("RDM:")
            PrintMatrix(rdm, Round = 4)
        w = np.sqrt(la.eigh(rdm@np.conj(rdm.T))[0])
        if debug:
            print(f"sqrt(rdm@ conj(rdm.T)):\n{w}")
        
        return np.log(np.sum(w))

    def save(self, name = None, tag = None, directory = None):
        if directory == None:
            directory = self.data_directory
        if name == None:
            name = "Defective_Ising_Model_N_"+str(self.N)+"with_"+str(self.boundaries)+"_BC"
            if tag != None:
                name = name + "_"+str(tag)

        
        path = directory+name+'/'
        if path[0] == '~':
            path = os.path.expanduser(path)
        output_dir = pathlib.Path(path)

        output_dir.mkdir(parents=True, exist_ok=True)

        #self.Single_Interval_Gamma_A_Partial_DF.to_pickle(path+"Single_Interval_Gamma_A_Partial_DF.pickle")    
        self.model_DF.to_pickle(path+"model_DF.pickle")   
        
        
        Ham_DF = pd.DataFrame(self.Ham)
        Ham_DF.to_pickle(path+"Ham_DF.pickle")

        Ham_Eigensystem_DF = pd.DataFrame(self.Ham_Eigensystem)
        Ham_Eigensystem_DF.to_pickle(path+"Ham_Eigensystem_DF.pickle")

        config_dict = {}
        config_dict['N'] = self.N
        config_dict['Jx'] = self.Jx
        config_dict['Jy'] = self.Jy
        config_dict['g'] = self.g
        config_dict['Ground_State_Energy'] = self.Ground_State_Energy
        config_dict['Gamma_A'] = self.Gamma_A
        config_dict['c_Fit'] = self.c_Fit
        config_dict['Num_for_Fit'] = self.Num_for_Fit
        config_dict["boundaries"] = self.boundaries
        config_dict['cutoff'] = self.cutoff

        config_DF = pd.DataFrame([config_dict])
        config_DF.to_pickle(path+"config_DF.pickle")

    def load(self, name = None, tag = None, directory = None, load_results_only = False):
        if directory == None:
            directory = self.data_directory
        if name == None:
            name = "Defective_Ising_Model_N_"+str(self.N)+"with_"+str(self.boundaries)+"_BC"
            if tag != None:
                name = name + "_"+str(tag)

        path = directory+name + '/'
        if path[0] == '~':
            path = os.path.expanduser(path)

        if not load_results_only:
            #self.Single_Interval_Gamma_A_Partial_DF = pd.read_pickle(path+"Single_Interval_Gamma_A_Partial_DF.pickle")
            
            Ham_DF = pd.read_pickle(path+"Ham_DF.pickle")
            self.Ham = Ham_DF.values

            Ham_Eigensystem_DF = pd.read_pickle(path+"Ham_Eigensystem_DF.pickle")
            self.Ham_Eigensystem = Ham_Eigensystem_DF.values
        
        self.model_DF = pd.read_pickle(path+"model_DF.pickle")
    

        config_DF = pd.read_pickle(path+"config_DF.pickle")
        config_dict = config_DF.T.to_dict()[0]

        self.N = config_dict['N']
        self.Jx = config_dict['Jx']
        self.Jy = config_dict['Jy']
        self.g = config_dict['g']
        self.Ground_State_Energy = config_dict["Ground_State_Energy"]
        self.Gamma_A = config_dict["Gamma_A"]
        self.c_Fit = config_dict["c_Fit"]
        self.Num_for_Fit = config_dict['Num_for_Fit']
        self.boundaries = config_dict['boundaries']
        self.cutoff = config_dict['cutoff']
            
class SpinChain:

    def __init__(self, N = 8, Jx = 1, Jy = 1, Jz  = 0, g = 0, boundaries = 'open', cutoff = 12, init_ham = False, init_gs = False):
        self.N = N
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.g = g
        self.boundaries = boundaries
        self.bc = self.boundaries
        self.cutoff = cutoff

        if isinstance(self.Jx, (np.floating, float, int)):
            self.Jx = self.Jx * np.ones(self.N)
        
        if isinstance(self.Jy, (np.floating, float, int)):
            self.Jy = self.Jy * np.ones(self.N)
        
        if isinstance(self.Jz, (np.floating, float, int)):
            self.Jz = self.Jz * np.ones(self.N)
        
        if isinstance(self.g, (np.floating, float, int)):
            self.g = self.g * np.ones(self.N)
        
        if not isinstance(self.Jx, np.ndarray):
            print(f"For Jx, Please uses either a single constant or a dim({str(self.N)}) array")
            return None
        if not isinstance(self.Jy, np.ndarray):
            print(f"For Jy, Please uses either a single constant or a dim({str(self.N)}) array")
            return None
        if not isinstance(self.Jz, np.ndarray):
            print(f"For Jz, Please uses either a single constant or a dim({str(self.N)}) array")
            return None
        
        if len(self.Jx) != self.N:
            print("Jx Array is length "+str(len(self.Jx))+". Please use length "+str(self.N))
            return None
        if len(self.Jy) != self.N:
            print("Jy Array is length "+str(len(self.Jy))+". Please use length "+str(self.N))
            return None
        if len(self.Jz) != self.N:
            print("Jz Array is length "+str(len(self.Jz))+". Please use length "+str(self.N))
            return None

        if not isinstance(self.g, np.ndarray):
            print("Please uses either a single constant or a dim("+str(self.N)+") array")
            return None
        if len(self.g) != self.N:
            print("Array is length "+str(len(self.g))+". Please use length "+str(self.N))
            return None

        self.Ham = None
        self.Ham_Eigensystem = None
        self.gs = None

        if init_ham:
            self.Ham_Builder
        
        self.rdm_dict = {}
        self.rdm_pt_dict = {}
        self.evs = {}

    def Ham_Builder(self, redo = False, Return = False):
    
        if self.Ham is not None and not redo:
            if Return:
                return self.Ham
        
        N = self.N
        Ham = np.zeros((2**N, 2**N), dtype=complex)

        # Building X Bit
        matrices = [0]*N
        if self.bc == "open":
            bonds = N-1

        if self.bc  == 'periodic':
            bonds = N
        
        for m in range(bonds):
            if self.Jx[m]!=0:
                sigma_m = Single_Site_Operator(sigma_x, Id, m % N, N)
                sigma_m_next = Single_Site_Operator(sigma_x, Id, (m+1)%N, N)
                Ham+= self.Jx[m]*np.matmul(sigma_m, sigma_m_next)
            if self.Jy[m] != 0:
                sigma_m = Single_Site_Operator(sigma_y, Id, m %N, N)
                sigma_m_next = Single_Site_Operator(sigma_y, Id, (m+1)%N, N)
                Ham+= self.Jy[m]*np.matmul(sigma_m, sigma_m_next)
            if self.Jz[m] != 0:
                sigma_m = Single_Site_Operator(sigma_z, Id, m %N, N)
                sigma_m_next = Single_Site_Operator(sigma_z, Id, (m+1)%N, N)
                Ham+= self.Jz[m]*np.matmul(sigma_m, sigma_m_next)

        for m in range(N):
            if self.g[m] != 0:
                Ham+= self.g[m]*Single_Site_Operator(sigma_z, Id, m, N)

        self.Ham = -1*Ham/2

        if Return:
            return self.Ham

    def Get_Ham_Eigensystem(self, redo = False, Return = False):
        if self.Ham is None:
            self.Ham_Builder(Return = False)
        if self.Ham_Eigensystem is None and not redo:
            if Return:
                return self.Ham_Eigensystem
        
        self.Ham_Eigensystem = la.eig(self.Ham)
        if Return:
            return self.Ham_Eigensystem
    
    def Two_Site_RDM(self, state, m1, m2, state_name = None, debug = False, Return = False, pt = 'no'):
        m1 = int(m1)%self.N
        m2 = int(m2)%self.N

        if (m1,m2, state_name) in self.rdm_dict and pt == 'no':
            if Return:
                return self.rdm_dict[(m1,m2, state_name)];
        
        if (m1,m2, state_name) in self.rdm_pt_dict and pt == 'yes':
            if Return:
                return self.rdm_dict[(m1,m2, state_name)];

        N = self.N
        if (m1, m2, state_name) not in self.evs:
            sigma_x_1 = Single_Site_Operator(sigma_x, Id, m1, N)
            sigma_x_2 = Single_Site_Operator(sigma_x, Id, m2, N)

            sigma_y_1 = Single_Site_Operator(sigma_y, Id, m1, N)
            sigma_y_2 = Single_Site_Operator(sigma_y, Id, m2, N)

            sigma_z_1 = Single_Site_Operator(sigma_z, Id, m1, N)
            sigma_z_2 = Single_Site_Operator(sigma_z, Id, m2, N)


            sigma_plus_1 = Single_Site_Operator((sigma_x+1j*sigma_y)/2., Id, m1, N) # need factors of 1/2  and i
            sigma_minus_2 = Single_Site_Operator((sigma_x-1j*sigma_y)/2., Id, m2, N) # need factors of 1/2 and i

            sigma_minus_1 = Single_Site_Operator((sigma_x-1j*sigma_y)/2., Id, m1, N) # need factors of 1/2  and i
            sigma_plus_2 = Single_Site_Operator((sigma_x+1j*sigma_y)/2., Id, m2, N) # need factors of 1/2 and i

            Identity = np.identity(2**N)
            evs = {}
            evs['p_x0'] = Expectation_Value(state, sigma_x_1)
            evs['p_0x'] = Expectation_Value(state, sigma_x_2)
            
            evs['p_y0'] = Expectation_Value(state, sigma_y_1)
            evs['p_0y'] = Expectation_Value(state, sigma_y_2)
            
            evs['p_z0'] = Expectation_Value(state, sigma_z_1)
            evs['p_0z'] = Expectation_Value(state, sigma_z_2)
            
            evs['p_xx'] = Expectation_Value(state, sigma_x_1@sigma_x_2)
            evs['p_xy'] = Expectation_Value(state, sigma_x_1@sigma_y_2)
            evs['p_xz'] = Expectation_Value(state, sigma_x_1@sigma_z_2)

            evs['p_yx'] = Expectation_Value(state, sigma_y_1@sigma_x_2)
            evs['p_yy'] = Expectation_Value(state, sigma_y_1@sigma_y_2)
            evs['p_yz'] = Expectation_Value(state, sigma_y_1@sigma_z_2)

            evs['p_zx'] = Expectation_Value(state, sigma_z_1@sigma_x_2)
            evs['p_zy'] = Expectation_Value(state, sigma_z_1@sigma_y_2)
            evs['p_zz'] = Expectation_Value(state, sigma_z_1@sigma_z_2)

            self.evs[(m1, m2, state_name)] = evs
        else:
            evs = self.evs[(m1, m2, state_name)]

        if debug:
            print(f"p_x0 = {evs['p_x0']}, p_0x = {evs['p_0x']}")
            print(f"p_y0 = {evs['p_y0']}, p_0y = {evs['p_0y']}")
            print(f"p_z0 = {evs['p_z0']}, p_0z = {evs['p_0z']}\n")
            print(f"p_xx = {evs['p_xx']}, p_xy = {evs['p_xy']}, p_xz = {evs['p_xz']}")
            print(f"p_yx = {evs['p_yx']}, p_yy = {evs['p_yy']}, p_yz = {evs['p_yz']}")
            print(f"p_zx = {evs['p_zx']}, p_zy = {evs['p_zy']}, p_zz = {evs['p_zz']}\n")

        a11 = 1+evs['p_0z']+evs['p_z0']+evs['p_zz']
        a21 = evs['p_0x']-1j*evs['p_0y']+evs['p_zx']-1j*evs['p_zy']
        a31 = evs['p_0x']-1j*evs['p_y0']+evs['p_xz']-1j*evs['p_yz']
        a41 = evs['p_xx']-1j*evs['p_yx']-1j*evs['p_xy']-evs['p_yy']

        a31_pt = evs['p_0x']+1j*evs['p_y0']+evs['p_xz']+1j*evs['p_yz']
        a41_pt = evs['p_xx']+1j*evs['p_yx']-1j*evs['p_xy']+evs['p_yy']

        a12 = evs['p_0x'] + 1j*evs['p_0y'] + evs['p_zx'] + 1j*evs['p_zy']
        a22 = 1+ evs['p_z0'] - evs['p_0z'] - evs['p_zz']
        a32 = evs['p_xx'] - 1j*evs['p_yx'] + 1j*evs['p_xy'] + evs['p_yy']
        a42 = evs['p_x0'] - 1j*evs['p_y0'] - evs['p_xz'] + 1j*evs['p_yx']

        a32_pt = evs['p_xx'] + 1j*evs['p_yx'] + 1j*evs['p_xy'] - evs['p_yy']
        a42_pt = evs['p_x0'] + 1j*evs['p_y0'] - evs['p_xz'] - 1j*evs['p_yx']

        a13 = evs['p_x0'] + 1j*evs['p_y0'] + evs['p_xz'] + 1j*evs['p_yz']
        a23 = evs['p_xx'] + 1j*evs['p_yx'] - 1j*evs['p_xy'] + evs['p_yy']
        a33 = 1 - evs['p_z0'] + evs['p_0z'] -evs['p_zz']
        a43 = evs['p_0x'] - 1j*evs['p_0y'] - evs['p_zx'] + 1j*evs['p_zy']

        a13_pt = evs['p_x0'] - 1j*evs['p_y0'] + evs['p_xz'] - 1j*evs['p_yz']
        a23_pt = evs['p_xx'] - 1j*evs['p_yx'] - 1j*evs['p_xy'] - evs['p_yy']

        a14 = evs['p_xx'] + 1j*evs['p_yx'] + 1j*evs['p_xy'] - evs['p_yy']
        a24 = evs['p_x0'] + 1j*evs['p_y0'] - evs['p_xz'] - 1j*evs['p_yz']
        a34 = evs['p_0x'] + 1j*evs['p_0y'] - evs['p_zx'] - 1j*evs['p_zy']
        a44 = 1 - evs['p_z0'] - evs['p_0z'] + evs['p_zz']

        a14_pt = evs['p_xx'] - 1j*evs['p_yx'] + 1j*evs['p_xy'] + evs['p_yy']
        a24_pt = evs['p_x0'] - 1j*evs['p_y0'] - evs['p_xz'] + 1j*evs['p_yz']

        if pt == 'no' or pt == 'both':
            self.rdm_dict[(m1,m2, state_name)] = np.array([[a11, a21, a31, a41],\
                                                           [a12, a22, a32, a42],\
                                                           [a13, a23, a33, a43],\
                                                           [a14, a24, a34, a44]]) /4.
            if Return:
                return self.rdm_dict[(m1,m2, state_name)]
        if pt == 'yes' or pt == 'both':
            self.rdm_pt_dict[(m1,m2, state_name)] = np.array([[a11,    a21,    a31_pt, a41_pt],\
                                                              [a12,    a22,    a32_pt, a42_pt],\
                                                              [a13_pt, a23_pt, a33,    a43],\
                                                              [a14_pt, a24_pt, a34,    a44]]) /4.
            if Return:
                return self.rdm_pt_dict[(m1,m2, state_name)]
                   
    def Reduced_Density_Matrix(self, interval, state,  debug = False):
        rho = np.outer(state, state)
        s_size = len(state)
        ss_size = 2**(len(interval))
        if debug:
            print(f'Trace of rho: {np.trace(rho)}')
            print(f'Trace of rho^2: {np.trace(rho@rho)}')
        
        rho_A = np.zeros((ss_size,ss_size))
        sites_B = np.setdiff1d(interval, np.arange(0,self.N))

        #for i,j in range(np.setdiff1d(np.arange(interval[0],interval[1]+1), np.arange([0,self.N+1]))):

    def Two_Site_Entropy(self, m1, m2, state = None, debug = False, state_name = None, mi = False):
        if state is None:
            if self.gs is None:
                if debug:
                    print(f'Need to calculate gs.')
                self.Charge_Sector_Ground_States(Return = False, debug = debug)
                if debug:
                    print(f'Calculated gs')
            if len(self.gs) == 1:
                state_name = 'gs'+str(list(self.gs.keys())[0])
                state = self.gs[list(self.gs.keys())[0]]
            
            if len(self.gs) != 1:
                print('No state specified and ground state is degenerate, please specify a state!')
                return None
        if debug:
            print(f"Sites = {{{m1},{m2}}}")
        RDM = self.Two_Site_RDM(state, m1, m2, debug = debug, Return = True, state_name = state_name)
        if debug:
            print("RDM:")
            PrintMatrix(RDM, Round = 4)
        eigs = la.eigh(RDM)[0]
        #print(eigs)
        S_arr = eigs*np.log(eigs)
        for i in np.arange(len(S_arr)):
            if np.isnan(S_arr[i]):
                S_arr[i] = 0
        S = -np.sum(S_arr)
        
        if mi:
            v1 = np.array([1,0])
            v2 = np.array([0,1])
            mat1 = np.kron(v1, np.identity(2))
            mat2 = np.kron(v2, np.identity(2))
            rdmA = mat1@RDM@mat1.T + mat2@RDM@mat2.T
            if debug:
                print(f'rdmA:')
                PrintMatrix(rdmA)

            w = la.eigh(rdmA)[0]
            if debug:
                print(f"Eigenvalues of the rdmA:\n{w}")

            SA = 0
            for eig in w:
                if debug:
                        print(f'current eig is {eig}')
                if np.abs(eig) > 10**(-1*self.cutoff):
                    if debug:
                        print(f'|Eig| is bigger than zero')
                    SA+= -1*eig*np.log(eig)

            mat1 = np.kron(np.identity(2), v1)
            mat2 = np.kron(np.identity(2), v2)
            rdmB = mat1@RDM@mat1.T + mat2@RDM@mat2.T
            if debug:
                print(f'rdmB:')
                PrintMatrix(rdmB)

            SB = 0
            for eig in w:
                if debug:
                        print(f'current eig is {eig}')
                if np.abs(eig) > 10**(-1*self.cutoff):
                    if debug:
                        print(f'|Eig| is bigger than zero')
                    SB+= -1*eig*np.log(eig)

            return [S, SA, SB, SA+SB-S]

        return S

    def Spin_Charge_Parity(self, state, debug = False, Round = 10, Round_Out = True):
        N = int(np.log2(len(state)))
        expectationValue =Expectation_Value(state, Spin_Charge_Operator(N), Round = Round)
        if debug:
            print(f'Expectation Value: {expectationValue}')
        charge = round(np.real(expectationValue))
        if debug:
            print(f'charge = {charge}')
        if not Round_Out:
            if debug:
                print(f'Returning {expectationValue}')
            return expectationValue
        if debug:
                print(f'Returning {charge}')
        return charge

    def Two_Site_Negativity(self, m1, m2, state = None, debug = False, state_name = None):
        if state is None:
            if self.gs is None:
                if debug:
                    print(f'Need to calculate gs.')
                self.Charge_Sector_Ground_States(Return = False, debug = debug)
                if debug:
                    print(f'Calculated gs')
            if len(self.gs) == 1:
                state_name = 'gs'+str(list(self.gs.keys())[0])
                state = self.gs[list(self.gs.keys())[0]]
            
            if len(self.gs) != 1:
                print('No state specified and ground state is degenerate, please specify a state!')
                return None

        rdm_pt = self.Two_Site_RDM(state, m1, m2, debug = debug, Return = True, pt = 'yes', state_name = state_name)

        if debug:
            print('rdm_pt')
            PrintMatrix(rdm_pt, Round = 4)
        
        eigs = np.sqrt(la.eigh(rdm_pt@np.conj(rdm_pt.T))[0])
        if debug:
            print(f'Eigs of sqrt(rdm_pt@rdm_pt^t): {eigs}')
            print(f'||rdm_pt||: {np.sum(eigs)}')
        return np.log(np.sum(eigs))

    def Charge_Sector_Ground_States(self, debug = False, Round = 10, Round_Out = True, Return = True):
        if self.Ham_Eigensystem is None:
            self.Get_Ham_Eigensystem()
        
        eig_sys = self.Ham_Eigensystem
        gs = {}
        if np.round(eig_sys[0][0], Round) != np.round(eig_sys[0][1], Round):
            if debug:
                print("Nondegenerate Ground State")
            gs[int(self.Spin_Charge_Parity(eig_sys[1][:,0]))] = eig_sys[1][:,0]
            self.gs = gs
            if Return:
                return self.gs
            if not Return:
                return None
        if debug:
            print("Degenerate Ground State")
        charge_mat = np.zeros((2,2))
        Q_Mat = Spin_Charge_Operator(self.N)
        print('1')
        print('2')
        for j in range(2):
            for k in range(2):
                print(f'(j,k) = ({j},{k})')
                if j == k:
                    state = eig_sys[1][:,j]
                    Q = np.conj(state)@Q_Mat@state
                    if debug:
                        print(f'    norm: {np.sum(np.conj(state)*state)}')
                        print(f'Q = {Q}')
                    charge_mat[j,k] = Q #Spin_Charge_Parity(state, Round_Out=Round_Out, debug = debug)
                if j!= k:
                    state = (eig_sys[1][:,j]+eig_sys[1][:,k]*np.sign(j-k))/np.sqrt(2)
                    Q = np.conj(state)@Q_Mat@state
                    if debug:
                        print(f'    norm: {np.sum(np.conj(state)*state)}')
                        print(f'Q = {Q}')
                    charge_mat[j,k] = Q
                    #charge_mat[j,k] = Spin_Charge_Parity(state, Round_Out=Round_Out, debug=debug)
        if debug:
            print('Charge Matrix:')
            PrintMatrix(charge_mat, Round = 4)

        charge_eig_sys = la.eig(charge_mat)
        if debug:
            print('Charge Matrix eigensystem')
            PrintEigensystem(charge_mat, Round = 4)

        for j in range(len(charge_eig_sys[0])):
            state = charge_eig_sys[1][0,j]*eig_sys[1][:,0]+charge_eig_sys[1][1,j]*eig_sys[1][:,1]
            state = state/np.sqrt(np.sum(np.conj(state.T)*state))
            gs[int(charge_eig_sys[0][j])] = state

        self.gs = gs
        if Return:
            return gs

def Save_Dict_of_Models(model_dict, dict_name, main_directory = "data/"):
    directory = main_directory+"/"+dict_name + "/"
    for key in model_dict:
        model_dict[key].save(name = "_"+str(key), directory = directory)
    
    labels = np.array(list(model_dict.keys()))
    np.savez(directory+"key.npz", labels)

def Save_List_of_Models(model_list, list_name, main_directory = "data/"):
    directory = main_directory+"/"+list_name + "/"
    for i in range(len(model_list)):
        model_list[i].save(name = "_"+str(i), directory = directory)
    
    labels = np.array(list(range(len(model_list))))
    np.savez(directory+"key.npz", labels)

def Load_Dict_of_Models(dict_name, main_directory = "data/", load_results_only = False):
    folder = dict_name + "/"
    directory = main_directory+folder
    
    if directory[0] == '~':
            directory = os.path.expanduser(directory)

    keys = np.load(directory+"key.npz", allow_pickle=True)['arr_0']
    model_dict = {}
    for key in keys:
            model_dict[key] = MajoranaFermionChain(load_name="_"+str(key),  load_results_only=load_results_only, data_directory=directory)
    
    return model_dict
        
def Load_List_of_Models(list_name, main_directory = "data/", load_results_only = False):
    folder = list_name + "/"
    directory = main_directory+folder
    
    if directory[0] == '~':
            directory = os.path.expanduser(directory)

    keys = np.load(directory+"key.npz", allow_pickle=True)['arr_0']
    model_list = []
    for key in keys:
            model_list.append(MajoranaFermionChain(load_name="_"+str(key),  load_results_only=load_results_only, data_directory=directory))
    
    return model_list

def Plot_Entanglement_Spectra(spectrum_list, markers = ["D"], facecolors = [None], X_axis = None, delta = 0.02, num_of_eigenvalues = 16, \
    dpi = 100, figsize = (10,5), cmaps = [cc.cm.glasbey], save = False, title = None, xlabel = None, ylabel = None, overlay_data = None, overlay_cmap = None, overlay_cutoff = None, markersize = 75, overlay_legend = None, data_legend = None, labels = None):
    
    if isinstance(spectrum_list[0][0], np.floating):
        spectrum_list = [spectrum_list]
    
    if type(X_axis) == list or type(X_axis) is np.ndarray:
        if isinstance(X_axis[0], np.floating):
            x_axis = [X_axis]

        if len(X_axis) < len(spectrum_list):
            print('Not Enough X_axis, Using all the Same')
            X_axis = [X_axis[0]]*len(spectrum_list)
    
    if not isinstance(markers, list):
        markers = [markers]
    
    if not isinstance(facecolors, list):
        facecolors = [facecolors]
    
    if not isinstance(cmaps, list):
        cmaps = [cmaps]
    
    if not isinstance(labels, list):
        labels = [labels]
    
    if len(markers) < len(spectrum_list):
        print('Not Enough Markers, Using all the Same')
        markers = [markers[0]]*len(spectrum_list)

    if len(facecolors) < len(spectrum_list):
        print('Not Enough facecolors, Using all the Same')
        facecolors = [facecolors[0]]*len(spectrum_list)
    
    if len(cmaps) < len(spectrum_list):
        print('Not Enough cmaps, Using all the Same')
        cmaps = [cmaps[0]]*len(spectrum_list)
    
    if len(labels) < len(spectrum_list):
        print('Not Enough labels')
        labels = list + [None]*(len(spectrum_list)-len(labels))

    fig = plt.figure(dpi = dpi, figsize=figsize)
    ax = fig.add_subplot(111)
    overlay_legends = []
    
    
    for j in range(len(spectrum_list)):
        print(f'Doing spectrum {j}')
        spectrums = spectrum_list[j]
        num = len(spectrums)
        cmap = cmaps[j]
        print(cmap)
        if type(cmap) is str:
            c = [cmap]*num
        else:
            c = cmap(np.linspace(0,1,num_of_eigenvalues))
        
        if X_axis is None:
            x_axis = np.arange(len(spectrums))
        
        else:
            x_axis = X_axis[j]
        
        dat_list = []
        for i in range(num):
            label = None
            if i == 0:
                label = labels[j]
            dat = spectrums[i]
            dat = np.sort((dat))[:num_of_eigenvalues]
            dat_list.append(dat)
            xnum =len(dat)
        
            x = x_axis[i] + delta*np.linspace(-1., 1, xnum)
            
            ax.scatter(x,dat, marker=markers[j], color=c, s = markersize, facecolor = facecolors[j], label = label)
    
    if X_axis is not None:
        ax.set_xticks(X_axis[0])

    #print("Checking Data")
    if overlay_data is not None:
        #print(type(overlay_data[0]))
        if type(overlay_data[0][0]) != list and type(overlay_data[0][0]) != np.ndarray:
            overlay_data = [overlay_data]
        #print("Checking cmap")
        if overlay_cmap is not None:
            if type(overlay_cmap) != list and type(overlay_cmap) != np.ndarray:
                overlay_cmap = [overlay_cmap]
        
        if overlay_cutoff is not None:
            if type(overlay_cutoff) != list and type(overlay_cutoff) != np.ndarray:
                overlay_cmap = [overlay_cutoff]
        
        #print("Checking cmap if cmap None with data")
        if overlay_data is not None and overlay_cmap is None:
            #print("assigning cmap to overlay_cmap")
            overlay_cmap = [cmap]

        if len(overlay_data)>len(overlay_cmap):
            #print("making constant overlay_cmap because too short")
            color_to_use = overlay_cmap[0]
            overlay_cmap = np.empty(len(overlay_data, dtype = object), dtype=object)
            overlay_cmap.fill(color_to_use)
    
    
        
        #print(f"length of overlay_data = {len(overlay_data)}")
        #print(f"overlay_cmap = {overlay_cmap}")
        #print(f"length of overlay_cmap = {len(overlay_cmap)}")


        
        #print("Hi")
        if overlay_cutoff is None:
            overlay_cutoff = np.empty(len(overlay_data), dtype=int)
            overlay_cutoff.fill(num_of_eigenvalues)
        for i in range(len(overlay_data)):
            #print(f"Doing Data {i}")
            color = overlay_cmap[i]
            if type(color) == str:
                c = np.empty(overlay_cutoff[i], dtype = object)
                c.fill(color) 
            else:
                c = color(np.linspace(0,1,overlay_cutoff[i]))
            
            for j in range(len(overlay_data[i])):
                #print(f"doing j = {j}")
                y = np.sort(overlay_data[i][j])
                if overlay_cutoff[i]<len(y):
                    y = y[:overlay_cutoff[i]]
                #print(len(y))
                x = x_axis[j] + delta*np.linspace(-1.,1, len(y))
                if len(c) > len(y): 
                    c = c[:len(y)]
    
                ax.scatter(x,y, marker = "+", color = c, s = markersize)

            if overlay_legend is not None:
                overlay_legends.append(plt.Line2D([0],[0], label = overlay_legend[i], color = c[0] , marker = '+', linewidth = 0))       

    if data_legend is not None:
        overlay_legends.append(plt.Line2D([0],[0], label = data_legend, marker="D", linewidth=0, markeredgecolor= "k", markerfacecolor="w"))       
        

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(overlay_legends)     

    lgd = ax.legend(handles=handles, bbox_to_anchor=(1.05, 1))
    ax.set_xlabel(xlabel)
    if ylabel == None:
        ylabel = r"Eigenvalues of $\mathcal{H}_V$"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if type(save) == str:
        print("Saving")
        plt.savefig(save, bbox_extra_artists=([lgd]), bbox_inches='tight')

    plt.show()

def Get_Central_Charge(x, y, N, pre_factor = 3, Print = True, Return = False ):
    if len(x) != len(y):
        print("x and y have different lengths. Try again.")
        return None

    
    E = lambda l, c, S0: (c/pre_factor)*np.log(N*np.sin(np.pi * l/N)/np.pi) + S0
    
    c_Fit = opt.curve_fit(E, x, y)
    if Print:
        print(f'c = {c_Fit[0][0]}, fit covariance = {c_Fit[1][0]} \n S0 = {c_Fit[0][1]}, fit covariance = {c_Fit[1][1]}')
    if Return:
        return c_Fit
    
def AnalyticEntropy(intervals, epsilon, N):
    n = len(intervals)//2
    entropy = 0
    for i in range(n):
        for j in range(n):
            b = intervals[2*i+1]
            a = intervals[2*j]
            x = np.sin(np.abs((b-a))*np.pi/N)
            entropy += np.log(np.abs(N*x/np.pi))
        for j in range(i):
            ai =  intervals[2*i]
            aj = intervals[2*j]
            bi =  intervals[2*i+1]
            bj = intervals[2*j+1]
            xa = np.sin(np.abs(ai - aj)*np.pi/N)
            xb = np.sin(np.abs(bi - bj)*np.pi/N)
            entropy+= -1*np.log(np.abs(xa*N/np.pi)) - np.log(np.abs(xb*N/np.pi))

    return (entropy - n*np.log(epsilon))/6.

def Get_List_of_Intervals(list_of_endpoints):
    lengths = []
    for i in range(len(list_of_endpoints)):
        if type(list_of_endpoints[i]) == np.ndarray or type(list_of_endpoints[i]) == list:
            lengths.append(len(list_of_endpoints[i]))
    
    for i in lengths:
        for j in lengths:
            if i != j:
                print('Lists are of different lengths, please try again')
                return None
    length = lengths[0]

    for i in range(len(list_of_endpoints)):
        if type(list_of_endpoints[i]) == float:
            list_of_endpoints[i] = int(list_of_endpoints[i])
        
        if type(list_of_endpoints[i]) == int:
            list_of_endpoints[i] = np.ones(length, dtype=int)*list_of_endpoints[i]

    intervals = []
    for i in range(len(list_of_endpoints[0])):
        interval = []
        for j in range(len(list_of_endpoints)):
            interval.append(list_of_endpoints[j][i])
        
        intervals.append(interval)
    
    return intervals

def Sanity_Check_Periodic(List_of_N):
    Ising_periodic_GS_Energy = []
    for N in List_of_N:
        model = MajoranaFermionChain(N = N, boundaries = 'periodic')
        Ising_periodic_GS_Energy.append(np.real(model.Get_Ground_State_Energy()))

    XX_periodic_GS_Energy = []
    for N in List_of_N:
        model = MajoranaFermionChain(N = N, boundaries = 'periodic', Jx = 1, Jy = 1, g = 0)
        XX_periodic_GS_Energy.append(np.real(model.Get_Ground_State_Energy()))

    E = lambda N, E0, E1, E2, E3, E4: E0*N + E1 + E2/N + E3/(N**2) + E4/(N**3)

    Ising_Fit = opt.curve_fit(E, List_of_N, Ising_periodic_GS_Energy)
    XX_Fit = opt.curve_fit(E, List_of_N, XX_periodic_GS_Energy, p0 = [-0.1, -0.1, -0.1, -0.1, -0.1], bounds = [-1,1])
    
    

    print(f'The Ground State Energy Fitted To E = E0 N + E1 + E2/N + E3/N^2 + E4/N^3:')
    print("-----------------------------------------------------------------------------------------------")
    print(f'For Ising:')
    print(f'    (E0, E1, E2, E3) = {Ising_Fit[0]}')
    print(f'    c = {-1*Ising_Fit[0][2]*6/np.pi} (should be 1/2)')
    
    print(f'For XX')
    print(f'    (E0, E1, E2, E3) = {XX_Fit[0]}')
    print(f'    c = {-1*XX_Fit[0][2]*6/np.pi} (should be 1)')
    return XX_periodic_GS_Energy

def Delta_S(r, L, debug = False, upper_bound = None):
    if upper_bound == None:
        upper_bound = L 
    '''From https://arxiv.org/pdf/2204.03601.pdf'''
    coth = lambda x: np.cosh(x)/np.sinh(x)
    integrand = lambda h: np.tanh(np.pi * h * r) * (coth(np.pi * h)-1)

    integral = inte.quad(integrand, 0, upper_bound)
    if debug:
        return integral
    
    return np.pi*r* integral[0]
    
def Get_Dict_Of_Many_Entropies(model_dict, intervals, entropy_method = "TwoPointCor", F = 'p_chain', sites = False):
    '''For entropy method, one can chose: TwoPointCor, From_ModHam and From_Beta.
    Current string options for F are: p_chain (periodic chain, no defect) and open_on_boundary for open chain with interval on the boundary.'''
    entropy_dict = {}
    key_list = list(model_dict.keys())

    for key in key_list:
        entropy_dict[key] = []
        for interval in intervals:

            if entropy_method == 'TwoPointCor':
                entropy_dict[key].append(model_dict[key].Entanglement_Entropy(interval, sites=sites))

            if entropy_method == 'From_ModHam':
                entropy_dict[key].append(Entropy_From_H(model_dict[key].Modular_Hamiltonian(interval, sites=sites)))
            
            if entropy_method == 'From_Beta':
                entropy_dict[key].append(Entropy_From_H(model_dict[key].Modular_Hamiltonian_From_Beta(interval, F = F)))
            
    return entropy_dict

def Entropy_Dict_Plotter(entropy_dicts, x, cmaps = cm.cool, special_tags = None, ax = None, dpi = 120, fig_size = (10,5), save = False, cbar_pad=0.02, cbar_shift=-0.09,legend_loc = 0, xlim = None, ylim = None):
    '''Current special tags:
        label: string
        zorder: int
        lw: line width. Can be int or lists of ints.
        ms: line width. Can be int or lists of ints.
        line_style: line style,  can be string or list of strings.
        cbar_ticks: boolian to display tick marks for that color bar.
    
    '''
    has_a_label = False
    if type(entropy_dicts) !=  list:
        entropy_dicts = [entropy_dicts]
    if type(cmaps) != list:
        cmaps = [cmaps]*len(entropy_dicts) 
    if type(special_tags) != list:
        special_tags = [special_tags]*len(entropy_dicts)
    
    if len(cmaps)<len(entropy_dicts):
        cmaps = [cmaps[0]]*len(entropy_dicts)

    keys_list = []
    for entropy_dict in entropy_dicts:
        keys_list.append(list(entropy_dict.keys()))

    cbar_arrs = []
    for keys in keys_list:
        cbar_arrs.append(np.array(keys))

    cmap_for_plots = []
    cmaps_for_colorbar = []
    for i in range(len(cbar_arrs)):
        cbar_arr = cbar_arrs[i]
        cmap = cmaps[i]
        cmap_for_plots.append(cmap(cbar_arr/max(cbar_arr)))
        norm = colors.Normalize(vmin = min(cbar_arr), vmax=max(cbar_arr))
        cmaps_for_colorbar.append(cm.ScalarMappable(cmap = cmap, norm = norm))

    
    if ax == None:
        fig, ax = plt.subplots(1, figsize = fig_size, dpi = dpi)
    
    for i in range(len(entropy_dicts)):
        keys = keys_list[i]
        entropy_dict = entropy_dicts[i]
        for j in range(len(keys)):
            key = keys[j]
            y = entropy_dict[key]

            color = cmap_for_plots[i][j]
            label = None
            zorder = None
            label = None
            line_styles = ['-']
            ms = None
            lw = None

            try:
                if j==0:
                    label = special_tags[i]['label']
                    has_a_label = True
            except:
                pass
            
            try:
                if j==0:
                    zorder = special_tags[i]['zorder']
            except:
                pass

            try:
                line_styles = special_tags[i]['line_style']
                if type(line_styles) is not list:
                    line_styles = [line_styles]
            except:
                pass
            
            try:
                ms = special_tags[i]['ms']
            except:
                pass
            
            try:
                lw = special_tags[i]['lw']
            except:
                pass
            
            for k in range(len(line_styles)):
                line_style = line_styles[k]
                
                if k!=0:
                    label = None
                ax.plot(x,y, line_style, ms = ms, lw = lw, color = color, zorder = zorder, label = label)

        shift = len(cbar_arrs)-1-i
        cbar = plt.colorbar(cmaps_for_colorbar[i], ax=ax, pad = cbar_pad+cbar_shift*shift)
        try:
            if not special_tags[i]['cbar_ticks']:
                cbar.set_ticks([])
        except:
            pass

        if has_a_label:                
            plt.legend(loc = legend_loc)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)  
        

    if type(save) == str:
        plt.savefig(save)

def Matrix_Plotter(matrix, cmap = cm.viridis, dpi = 120, figsize = (5,5), xlabels = 'n', ylabels = 'm', xticks = True, yticks = True, colorbar = True, magnitude_phase = False, same_scale = True, center_zero = False, min_max_arr = None, title = None, titlesize = 16, phase_cmap = cm.hsv, save = False, cbarscale = None, linthresh = 0.1, Return = False, h_lines = [], v_lines = [], Return_Params = False, xtick_vals = None, ytick_vals = None, xtick_labels = None, ytick_labels = None, subplot_adjust={}, phase_alpha = False):
    is_complex = False
    if matrix.dtype == complex:
        is_complex = True

    if type(linthresh)!= list:
        linthresh = [linthresh,linthresh]
    
    cbarscale = [cbarscale,cbarscale]
    if not is_complex:
        fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
        axs = [ax]
        
        if min_max_arr==None:
            cbar_max = [matrix.max()]
            cbar_min = [matrix.min()]
            if center_zero:
                cbar_max = [max([abs(cbar_max[0]), abs(cbar_min[0])])]
                cbar_min = [-cbar_max[0]]
        
        else:
            cbar_min = [min_max_arr[0]]
            cbar_max = [min_max_arr[1]]

        cbar_ticks = [None]
        cbar_tick_labels = [None]


    if is_complex:
        figsize =(figsize[0], figsize[1])
        
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=figsize, dpi = dpi, constrained_layout = True)
        
    fig.suptitle(title, fontsize = titlesize, ha= 'center')
    matrices = [np.real(matrix)]
    labels = [None]
    cmaps = [cmap, cmap]
    if is_complex:
        if not magnitude_phase:
            labels = ["Real", "Imaginary"]
            matrices.append(np.imag(matrix))

            if min_max_arr == None:
                cbar_max_Real=matrices[0].max()
                cbar_min_Real=matrices[0].min()
                cbar_max_Imag=matrices[1].max()
                cbar_min_Imag=matrices[1].min()

                if center_zero:
                    cbar_abs_max_Real = max([abs(cbar_max_Real),abs(cbar_min_Real)])
                    cbar_abs_max_Imag = max([abs(cbar_max_Imag),abs(cbar_min_Imag)])

                    cbar_max_Real=cbar_abs_max_Real
                    cbar_min_Real=-cbar_abs_max_Real
                    cbar_max_Imag=cbar_abs_max_Imag
                    cbar_min_Imag=-cbar_abs_max_Imag
                
                if same_scale:
                    cbar_max_Real = max([cbar_max_Real,cbar_max_Imag])
                    cbar_min_Real = min([cbar_min_Real,cbar_min_Imag])
                    cbar_max_Imag = cbar_max_Real
                    cbar_min_Imag = cbar_min_Real

            else:
                cbar_min_Real = min_max_arr[0]
                cbar_max_Real = min_max_arr[1]
                if same_scale or (len(min_max_arr)==2):
                    cbar_min_Imag = min_max_arr[0]
                    cbar_max_Imag = min_max_arr[1]
                else:
                    cbar_min_Imag = min_max_arr[2]
                    cbar_max_Imag = min_max_arr[3]
                

            cbar_max = [cbar_max_Real, cbar_max_Imag]
            cbar_min = [cbar_min_Real, cbar_min_Imag]
            
            cbar_ticks = [None, None]
            cbar_tick_labels = [None, None]

        if magnitude_phase:
            labels = ["Magnitude", "Phase"]
            matrices = [matrix.__abs__(), np.angle(matrix)]
            cmaps = [cmap, phase_cmap]
            
            if min_max_arr == None:
                cbar_max_mag = matrices[0].max()
                cbar_min_mag = matrices[0].min()
            
            else:
                cbar_min_mag = min_max_arr[0]
                cbar_max_mag = min_max_arr[1]
            cbar_max = [cbar_max_mag, np.pi]
            cbar_min = [cbar_min_mag, -np.pi]
            
            cbar_ticks = [None, [-np.pi, 0, np.pi]]
            cbar_tick_labels = [None, [r'$-\pi$', r'$0$', r'$\pi$']]
            cbarscale[1] = None


    if type(xlabels) == 'str' or xlabels == None:
        xlabels = [xlabels]*len(axs)
    if type(ylabels) == 'str' or ylabels == None:
        ylabels = [ylabels]*len(axs)
    if len(xlabels)<len(axs):
        xlabels = [xlabels[0]]*len(axs)
    if len(ylabels)<len(axs):
        ylabels = [ylabels[0]]*len(axs)   
    cbars = []
    for i in range(len(axs)):
        
        if cbarscale[i] == 'symlog':
            im = axs[i].imshow(matrices[i], cmap = cmaps[i], norm=colors.SymLogNorm(linthresh[i], vmin=cbar_min[i], vmax=cbar_max[i]))    
        
        else:
            alpha = None
            if magnitude_phase and i == 1 and phase_alpha:
                alpha = np.abs(matrices[0])
                if cbarscale[0] == 'symlog':
                    alpha[alpha>linthresh[0]] = np.log10(alpha[alpha>linthresh[0]])+linthresh[0]

                alpha *= 1/alpha.max()
                

            im = axs[i].imshow(matrices[i], cmap = cmaps[i], vmin = cbar_min[i], vmax = cbar_max[i], norm=cbarscale[i], alpha = alpha)

        xtick_params = {'top':xticks, 'bottom':False, 'labeltop':xticks, 'labelbottom':False}
        ytick_params = {'top':yticks, 'bottom':False, 'labeltop':yticks, 'labelbottom':False}

        axs[i].tick_params('x', **xtick_params)
        axs[i].tick_params('y', **ytick_params)

        axs[i].set_xlabel(xlabels[i])
        axs[i].set_ylabel(ylabels[i])
        axs[i].xaxis.set_label_position('top') 
        axs[i].set_title(labels[i])
        cbar = plt.colorbar(im, ax = axs[i], fraction=0.046, pad=0.04, ticks=cbar_ticks[i])
        if cbar_tick_labels[i]!=None:
            cbar.ax.set_yticklabels(cbar_tick_labels[i])
        
        x_tick_array = axs[i].get_xticks()
        
        x_tick_array = x_tick_array[x_tick_array%1==0].astype(int)
        x_tick_array = x_tick_array[x_tick_array>=0]
        x_tick_array = x_tick_array[x_tick_array<(matrices[i].shape[1])]
        
        axs[i].set_xticks(x_tick_array)

        y_tick_array = axs[i].get_yticks()
        y_tick_array = y_tick_array[y_tick_array%1==0].astype(int)
        y_tick_array = y_tick_array[y_tick_array>=0]
        y_tick_array = y_tick_array[y_tick_array<(matrices[i].shape[0])]
        axs[i].set_yticks(y_tick_array)

        if xtick_vals is not None:
            axs[i].set_xticks(xtick_vals, xtick_labels)
        if ytick_vals is not None:
            axs[i].set_yticks(ytick_vals, ytick_labels)
        
        cbars.append(cbar)
        
        for h_line in h_lines:
            line = h_line.copy()
            y = line['loc']
            line.pop('loc') 
            print(y)
            axs[i].axhline(y=y, xmin = 0, xmax = len(matrix), **line)
        for v_line in v_lines:
            line = v_line.copy()
            x = line['loc']
            line.pop('loc')
            axs[i].axvline(x=x, ymin = 0, ymax = len(matrix), **line)
    
    if Return_Params and len(axs)==1:
        plt.close(fig)
        return {'tick_array':[x_tick_array, y_tick_array] , 'tick_params':[xtick_params, ytick_params], 'cmap': cmap, 'cbar_bounds': [cbar_min[0], cbar_max[0]], 'cbar_ticks':cbar_ticks[0], 'cbar_tick_labels':cbar_tick_labels[0], 'cbarscale':cbarscale[0], 'symlogstart':symlogstart[0]}

    plt.subplots_adjust(**subplot_adjust)
    if type(save)==str:
        plt.savefig(save, bbox_inches = 'tight')

    if Return:
        return [fig, ax, cbars]

    if not Return:
        plt.show()

def FourPointFunction(a,b,c,d, RestrictedPropagator):
    cor_mat = np.zeros((4,4), dtype=complex)
    points = [a,b,c,d]
    for i in range(4):
        for j in range(i):
            if j!=i:
                cor_mat[i,j] = RestrictedPropagator[points[i]%len(RestrictedPropagator),points[j]%len(RestrictedPropagator)]
                cor_mat[j,i] = -cor_mat[i,j] #RestrictedPropagator[points[j],points[i]]

    #MFC.PrintMatrix(cor_mat)
    #print(cor_mat[0,1]*cor_mat[2,3] - cor_mat[0,2]*cor_mat[1,3]+cor_mat[1,2]*cor_mat[0,3])HAHA
    return pf.pfaffian(cor_mat)

def G_ab(Operators, RestrictedPropagator, Operator_Coefficients = [], PrintMat = False, debug = False):
    size = len(Operators)
    if len(Operator_Coefficients)!= len(Operators):
        Operator_Coefficients = np.ones(len(Operators))

    G = np.zeros((size,size), dtype=RestrictedPropagator.dtype)
    G_text = np.empty((size,size), dtype='U100')
    for index_i in range(size):
        for index_j in range(size):
            if debug: print(f'(index_i, index_j) = {index_i, index_j}')
            
            op_i = Operators[index_i]
            op_j = Operators[index_j]
            #print(f'op_i: {op_i}, type(op_i[0]): {type(op_i[0])}\nop_j: {op_j}, type(op_j[0]): {type(op_j[0])}\n-----')
            if not isinstance(op_i[0], (list, np.ndarray)):
            #    print('meow')
                op_i = [op_i]
            if not isinstance(op_j[0], (list, np.ndarray)):
            #    print('meow\n-----')
                op_j = [op_j]
            #print(f'op_i: {op_i}, type(op_i[0]): {type(op_i[0])}\nop_j: {op_j}, type(op_j[0]): {type(op_j[0])}\n-----')
            op_i_temp = op_i
            op_i = []
            op_j_temp = op_j
            op_j = []
            [op_i.append(x) for x in op_i_temp if x not in op_i]
            [op_j.append(x) for x in op_j_temp if x not in op_j]
            if debug: print(f'op_i: {op_i}\nop_j: {op_j}\n--------------------------------------------------------------------')
            
            AntiCommutator_ij = 0
            AntiCommutator_ji = 0
            for i in range(len(op_i)):
                for j in range(len(op_j)):
                    AntiCommutator_ij += FourPointFunction(op_i[i][0], op_i[i][1], op_j[j][0], op_j[j][1], RestrictedPropagator)
                    AntiCommutator_ji += FourPointFunction(op_j[j][0], op_j[j][1], op_i[i][0], op_i[i][1], RestrictedPropagator)

            expectation_i = 0
            expectation_j = 0
            for i in range(len(op_i)):
                expectation_i += RestrictedPropagator[op_i[i][0]%len(RestrictedPropagator), op_i[i][1]%len(RestrictedPropagator)]
                
            for j in range(len(op_j)):
                expectation_j += RestrictedPropagator[op_j[j][0]%len(RestrictedPropagator), op_j[j][1]%len(RestrictedPropagator)]


            string = '{'+str(op_i)+', '+str(op_j)+'}'
            G_text[index_i, index_j] = string
            
            G[index_i,index_j] = (0.5*(AntiCommutator_ij+AntiCommutator_ji) - expectation_i*expectation_j)#/(len(op_i)*len(op_j))**0.5

            if debug:
                print(f'G[{index_i},{index_j}] = {G[index_i,index_j]}')
    if PrintMat:
        print(isinstance(G_text[0,0], str))
        PrintMatrix(G_text)
    return -G

def Entanglement_Spectrum_From_H(H, max_number_of_eignvalues = 100, debug = False):
    eigs = la.eigh(H)[0]
    pos_eigs = np.sort(eigs[eigs>0])

    out = []
    number_of_eigenvalues = max_number_of_eignvalues
    if number_of_eigenvalues>2**len(pos_eigs):
        number_of_eigenvalues = 2**len(pos_eigs)
    if debug: print(f'Eigs: {pos_eigs}')
    if debug: print(f'ss_size = {len(pos_eigs)}')

    for i in range(number_of_eigenvalues):
        mask = (np.array([bool(int(n)) for n in np.binary_repr(i, width = len(pos_eigs))]))[::-1]
        out.append(np.sum(np.sum(pos_eigs[(mask)])))
    out = np.array(out)

    # for i in range(number_of_eigenvalues):
    #     mask = (np.array([bool(int(n)) for n in np.binary_repr(i, width = len(pos_eigs))]))[::-1]
    #     if debug: print(f'  i = {i}\n       mask: {mask}')
    #     eigenval = np.sum(eigs[np.array([np.logical_not(mask), mask])])
    #     if debug: print(f'      eigenval = {eigenval}')
    #     out.append(eigenval)
    
    # out = np.array(out)
    return np.sort(out)
    
def Energy_Variance(G, H, debug = False, relative = False, return_H_and_HH = False):
    ev_H = np.sum(H*G)

    ev_HH = 0
    for m in range(len(G)):
        for n in range(m):
            for p in range(len(G)):
                for q in range(p):
                    scale_mn = 1
                    scale_pq = 1
                    if m!=n:
                        scale_mn = 2
                    if p!=q:
                        scale_pq = 2
                    ev_HH += H[m,n]*H[p,q]*FourPointFunction(m,n,p,q, G)*scale_mn*scale_pq
    
    if debug: print(f'ev_H: {ev_H}, ev_H^2: {ev_H**2}, ev_HH = {ev_HH}')
    
    if return_H_and_HH:
        return [ev_H, ev_HH]
    var = ev_HH - ev_H**2
    if relative:
        var = ev_HH/(ev_HH**2) - 1
    return var

def G_ees(Gamma_A_Partial, I=['None'], use_mp = False, return_as_ndarray = True):
    if not isinstance(I, (list, np.ndarray)):
        I = [I]
    L = len(Gamma_A_Partial)
    if type(Gamma_A_Partial)==mpm.matrix:
        use_mp = True
    if not use_mp:
        schur = la.schur(Gamma_A_Partial)


        Gamma_C = schur[0].copy()
        Gamma_C[Gamma_C<-10e-11] = -1
        Gamma_C[Gamma_C>10e-11] = 1
        for i in I:
            if i != 'None':
                Gamma_C[2*i, 2*i+1] *= -1
                Gamma_C[2*i+1, 2*i] *= -1
        
        G_new = (np.eye(len(Gamma_C))+1j*(schur[1]@Gamma_C@schur[1].T))/2
    
    if use_mp:
        schur = mpm.mp.schur(Gamma_A_Partial)

        Gamma_C = mpm.mp.eye(L)
        for i in range(L):
            if mpm.mp.sign(mpm.mp.im(schur[1][i,i]))==-1:
                Gamma_C[i,i]*= -1

        pos_diags = []
        neg_diags = []
        for i in range(L//2):
            pos_diags.append(mpm.im(schur[1][L//2-i, L//2-i]))
            neg_diags.append(mpm.im(schur[1][i,i]))

        pos_diags = np.array(pos_diags)
        pos_diags_argsort = pos_diags.argsort()

        neg_diags = np.array(neg_diags)
        neg_diags_argsort_flipped = neg_diags.argsort()[::-1]


        for i in I:
            if i != 'None':
                i_pos = pos_diags_argsort[i]+L//2
                i_neg = neg_diags_argsort_flipped[i]
                Gamma_C[i_pos,i_pos]*=-1
                Gamma_C[i_neg, i_neg]*= -1
        G_new = (mpm.mp.eye(L)-schur[0]@Gamma_C@schur[0].transpose_conj())/2
        if return_as_ndarray:
            G_new = np.array(G_new,dtype='complex').reshape((L,L))
    return G_new

def Entropy_From_G(G, interval, sites = False, debug = False, cutoff = 14):
    Gamma_A = -1j*(G-np.eye(len(G)))
    N = len(G)//2
    if sites:
        site_list = interval
        interval = []
        for site in site_list:
            interval.append(site)
            interval.append(site)
        
    interval = np.array(interval).astype(int)
    xi_arr = interval[::2]%(N)
    xf_arr = interval[1::2]%(N)
    if debug:
        print(f' Starting Points {xi_arr}')
        print(f'End Points: {xf_arr}')
    
    mask = np.zeros((2*N,2*N), dtype=bool)
    false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
    for i in range(len(xi_arr)):
        xi = xi_arr[i]
        xf = xf_arr[i]
        index_arr = np.indices((2*N,2*N))
        tf_start = index_arr >= (2*xi)
        if xi >= 0:
            tf_start = ~tf_start
        tf_end = index_arr < 2*(xf+1)
        or_xy = np.logical_xor(tf_start,tf_end)
        mask_i = np.logical_and(or_xy[0],or_xy[1])
        if debug:
            print(f"Mask for interval {i+1}")
            PrintMatrix(np.int8(mask_i))
        mask = np.logical_or(mask, mask_i)
    
    false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
        
    if debug :
        print("Mask:")
        PrintMatrix(np.int8(mask))


    Gamma_A_Partial = np.delete(np.delete(Gamma_A,false_loc,axis=0), false_loc, axis = 1)

    Bipartite_S = lambda x: -x*np.log(x) - (1-x)*np.log(1-x)


   
    if Gamma_A_Partial.size == 0:
        return 0
    
    else: 
        
        ReEigs = np.round(la.eigh(-1j*Gamma_A_Partial)[0], cutoff)
        
        
        if debug:
            print("Eigenvalues: "+str(ReEigs))
        x_arr = (1+np.abs(ReEigs))/2      
        if debug:
            print("Nus: "+str(x_arr))
        
        Sl_arr = Bipartite_S(x_arr)
        
        for i in range(len(Sl_arr)):
            if np.isnan(Sl_arr[i]):
                Sl_arr[i] = 0
        
        if debug:
            print(f'Sl_arr:\n{Sl_arr}')
            
        S = np.round(np.sum(Sl_arr)/2,cutoff)

    return S
class MajoranaFermionChain_MPM:
    def __init__(self, N = 20, Jx = 1, Jy = 0, g = 1, b_sigma = 0, Q = 1, boundaries = "periodic", cutoff = 16, load_name = None, load_results_only = False, data_directory = "data/"):
        self.base_model = MajoranaFermionChain(N = N, Jx = Jx, Jy = Jy, g = g, b_sigma = b_sigma, Q = Q, boundaries = boundaries, cutoff = cutoff, load_name = load_name, load_results_only = load_results_only, data_directory = "data/")
        self.data_directory = data_directory
        self.load_results_only = load_results_only
        self.N = N
        self.Jx = Jx
        self.Jy = Jy
        self.g = g
        self.boundaries = boundaries
        self.cutoff = cutoff
        self.b_sigma = b_sigma # Topological Defect Term: EQ 2 of https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.128.090603
        self.Q = Q
        
        

        if type(load_name) == str:
            self.load(load_name, directory = self.data_directory, load_results_only = self.load_results_only)
        
        
        if isinstance(self.Jx, (np.floating, float, int)):
            self.Jx = self.Jx * np.ones(self.N)
        
        if isinstance(self.Jy, (np.floating, float, int)):
            self.Jy = self.Jy * np.ones(self.N)
        
        if isinstance(self.g, (np.floating, float, int)):
            self.g = self.g * np.ones(self.N)
        
        if isinstance(self.b_sigma, (np.floating, float, int)):
            self.b_sigma = self.b_sigma * np.ones(self.N)
        
        if not isinstance(self.Jx, np.ndarray):
            print(f"For Jx, Please uses either a single constant or a dim({str(self.N)}) array")
            return None
        if not isinstance(self.Jy, np.ndarray):
            print(f"For Jy, Please uses either a single constant or a dim({str(self.N)}) array")
            return None
        if len(self.Jx) != self.N:
            print("Jx Array is length "+str(len(self.Jx))+". Please use length "+str(self.N))
            return None

        if len(self.Jy) != self.N:
            print("Jy Array is length "+str(len(self.Jy))+". Please use length "+str(self.N))
            return None

        if not isinstance(self.g, np.ndarray):
            print("Please uses either a single constant or a dim("+str(self.N)+") array")
            return None
        if len(self.g) != self.N:
            print("Array is length "+str(len(self.g))+". Please use length "+str(self.N))
            return None

        if not isinstance(self.b_sigma, np.ndarray):
            print("Please uses either a single constant or a dim("+str(self.N)+") array")
            return None
        if len(self.b_sigma) != self.N:
            print("Array is length "+str(len(self.b_sigma))+". Please use length "+str(self.N))
            return None

        self.Ham = None
        self.Ham_Eigensystem = None
        self.Gamma_A = {'filled':None, 'flipped': None, 'empty':None}

        self.G = None
    
        self.H_E = {}

        self.schur_A = None

        # EE = Entanglement Entropy
        # ES = Entanglement Spectrum
        # FLN = Fermionic Logarithmic Negativity
        # BLN = Bosonic Logarithmic Negativity
        # GAPRE = Gamma_A_Partial_ReEigenvalues
        # GAPTE = Gamma_A_Partial partially transposed eigenvalues
        self.columns = ["GAPRE", "GAPTE", "ES", "EE", "FLN", "BLN"]
        self.model_DF = pd.DataFrame(columns = self.columns, dtype=object)
        self.ExpectationValues = {}
    
    def Ham_Builder(self):
        self.base_model.Ham_Builder()
        self.Ham = mpm.mp.matrix(self.base_model.Ham)

    
    def Get_Gamma_A(self, Return = True, debug = False, redo = False, large_Sort_warning = 1000, debug_round = 4, Full_Debug = False, zero_modes = 'filled', cutoff = 10**-10, zero_mode_mat = None):
        if self.Gamma_A[zero_modes] is not None and not redo:
            if Return:
                return self.Gamma_A[zero_modes]
            else:
                return None
        if self.Ham is None:
            self.Ham_Builder()

        A = -2j*self.Ham
        
        if self.schur_A is None: 
            W, B = mpm.mp.schur(A)
            self.schur_A = [W,B]
        
        else:
            W = self.schur_A[0]
            B = self.schur_A[1]

        B_Array = np.empty((2*self.N, 2*self.N), dtype=type(A[0,0]))
        W_Array = np.empty((2*self.N, 2*self.N), dtype=type(A[0,0]))

        for i in range(2*self.N):
            for j in range(2*self.N):
                B_Array[i,j] = B[i,j]
                W_Array[i,j] = W[i,j]
        
        if debug:
            print(f'B:\n')
            PrintMatrix(MPM_Mat_To_NP(B))
            print(f'\nW:\n')
            PrintMatrix(MPM_Mat_To_NP(W))


        diag = []
        for i in range(len(B_Array)):
            diag.append(mpm.im(B_Array[i, i]))
        
        diag = np.array(diag, dtype = type(diag[0]))
        idx = np.argsort(diag)

        B_Array = B_Array[idx,:][:, idx]
        W_Array = W_Array[:, idx]

        for i in range(self.N-1):
            
            temp_B = B_Array[:, -1].copy()
            temp_W = W_Array[:, -1].copy()

            B_Array[:, 2*(i+1):] = B_Array[:, 2*i+1:-1]
            W_Array[:, 2*(i+1):] = W_Array[:, 2*i+1:-1]

            temp_B = np.roll(temp_B, 2*(i+1))

            B_Array[:, 2*i+1] = temp_B
            W_Array[:, 2*i+1] = temp_W

            B_Array[:,2*(i+1):] = np.roll(B_Array[:,2*(i+1):], 1, axis = 0)
            
            if Full_Debug and debug:
                print(f'Did i = {i}:')
                PrintMatrix(MPM_Mat_To_NP(mpm.matrix(B_Array)))


        invsqrt2_mpm = 1/mpm.sqrt(2)

        transformation_matrix = la.block_diag(*[np.array([[-1j*invsqrt2_mpm, 1j*invsqrt2_mpm], [invsqrt2_mpm,invsqrt2_mpm]])]*self.N)
        B_Array = transformation_matrix@B_Array@np.conjugate(transformation_matrix.T)
        W_Array = W_Array@np.conjugate(transformation_matrix.T)

        occupation_matrices = [np.array([[0,-1],[1,0]])]*self.N
        if mpm.fabs(B_Array[-1, -2]) < cutoff:
            if zero_mode_mat is not None:
                occupation_matrices[-1] = zero_mode_mat
            if zero_mode_mat is None:
                if zero_modes == 'filled':
                    zero_mode_mat = np.array([[0,-1],[1,0]])
                if zero_modes == 'flipped':
                    zero_mode_mat = -np.array([[0,-1],[1,0]])
                if zero_modes == 'both':
                    zero_mode_mat = np.array([[0,1],[1,0]])
                if zero_modes == 'empty':
                    zero_mode_mat = np.array([[0,0],[0,0]])
                
                occupation_matrices[-1] = zero_mode_mat
                
        Gamma_B = la.block_diag(*occupation_matrices)

        W_Array_Right = np.conjugate(W_Array.copy())

        W_Array_Left = W_Array.copy()

        if debug:
            print(f'\n B_Array Matrix After Sorting:')
            PrintMatrix(MPM_Mat_To_NP(mpm.matrix(B_Array)))


        if mpm.fabs(B_Array[-1, -2]) < cutoff:
            found_correct_factor = False
            for factor in [-1, -1j, 1j, 1]:
                W_Array_Right_Temp = W_Array_Right.copy()
                if debug:
                    print(f'Doing Factor: {factor}')    
                W_Array_Right_Temp[:, 0:-2] = W_Array_Right_Temp[:, 0:-2]
                W_Array_Right_Temp[:, -2:] = np.conjugate(W_Array_Right_Temp[:, -2:])*factor # 13 needs 1j, 14 needs -1j, 15 needs -1j, 16 needs -1, 17 needs 1j, 18 needs -1j, 19 needs 1j

                Gamma_A_Array = W_Array_Left@Gamma_B@(W_Array_Right_Temp.T)
                Gamma_A = mpm.matrix(Gamma_A_Array)

                Gamma_A_NP_Array = MPM_Mat_To_NP(Gamma_A)
                if debug:
                    print(f'Difference Norm: {la.norm(Gamma_A_NP_Array - self.base_model.Get_Gamma_A(Return = True, zero_modes = zero_modes), ord = 1)}')
                if la.norm(Gamma_A_NP_Array - self.base_model.Get_Gamma_A(Return = True, zero_modes = zero_modes), ord = 1) < cutoff:
                    if debug:
                        print(f'Found Correct Factor! {factor}')
                    break

        else:   
            Gamma_A_Array = W_Array_Left@Gamma_B@(W_Array_Right.T) 
            Gamma_A = mpm.matrix(Gamma_A_Array)
        
        self.Gamma_A[zero_modes] = Gamma_A.copy()

        if Return:
            return Gamma_A


    def Get_G(self, Return = True, zero_modes = 'filled'):
        self.Get_Gamma_A(Return = None, zero_modes=zero_modes)
        if self.G is None:
            self.G = (mpm.mp.eye(len(self.Gamma_A[zero_modes]))+1j*self.Gamma_A[zero_modes])/2
            if Return:
                return self.G
        if Return:
            return self.G
    
    def Get_Gamma_A_Partial(self, interval, sites = False, debug = False, Redo = False, large_Sort_warning = 1000, debug_round = 4, zero_modes = 'filled', print_mask = False, Full_Debug = False):
            if sites:
                site_list = interval
                interval = []
                for site in site_list:
                    interval.append(site)
                    interval.append(site)
                
            interval = np.array(interval).astype(int)
            xi_arr = interval[::2]%(self.N)
            xf_arr = interval[1::2]%(self.N)
            if debug:
                print(f' Starting Points {xi_arr}')
                print(f'End Points: {xf_arr}')
            
            if self.Gamma_A[zero_modes] is None:
                self.Get_Gamma_A(Return = False, zero_modes=zero_modes, redo=Redo, large_Sort_warning=large_Sort_warning, debug=debug, Full_Debug=Full_Debug)

            Gamma_A_Array = np.empty((2*self.N, 2*self.N), dtype = type(self.Gamma_A[zero_modes][0,0]))
            for i in range(len(self.Gamma_A[zero_modes])):
                for j in range(len(self.Gamma_A[zero_modes])):
                    Gamma_A_Array[i,j] = self.Gamma_A[zero_modes][i,j]

            mask = np.zeros((2*self.N,2*self.N), dtype=bool)
            false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
            for i in range(len(xi_arr)):
                xi = xi_arr[i]
                xf = xf_arr[i]
                index_arr = np.indices((2*self.N,2*self.N))
                tf_start = index_arr >= (2*xi)
                if xi >= 0:
                    tf_start = ~tf_start
                tf_end = index_arr < 2*(xf+1)
                or_xy = np.logical_xor(tf_start,tf_end)
                mask_i = np.logical_and(or_xy[0],or_xy[1])
                if debug:
                    print(f"Mask for interval {i+1}")
                    PrintMatrix(np.int8(mask_i))
                mask = np.logical_or(mask, mask_i)
            
            false_loc = np.argwhere(np.all(mask[...,:]==False, axis = 0))
                
            if debug or print_mask:
                print("Mask:")
                PrintMatrix(np.int8(mask))

            Gamma_A_Partial = mpm.matrix(np.delete(np.delete(Gamma_A_Array,false_loc,axis=0), false_loc, axis = 1))

            return Gamma_A_Partial
        
    def Get_E_H(self, interval, Return = True, zero_modes = 'filled', redo = False, sites = False):
        key = str([interval, zero_modes])
        if key in self.H_E.keys() and not redo:
            if Return:
                return self.H_E[key]
        
        Gamma_A_Partial = self.Get_Gamma_A_Partial(interval, zero_modes=zero_modes, sites=sites)
        G_Partial = (mpm.mp.eye(len(Gamma_A_Partial))+1j*Gamma_A_Partial)/2 #self.Get_G(zero_modes=zero_modes)[2*interval[0]:2*(interval[1]+1), 2*interval[0]:2*(interval[1]+1)]
        U, G_Partial_tilde = mpm.mp.schur(G_Partial)

        B_egs = mpm.mp.eye(len(G_Partial_tilde))
        for i in range(len(B_egs)):
            #print(f'Doing {i}')
            #print(G_Partial_tilde[i,i])
            B_egs[i,i] = -mpm.mp.log(mpm.mp.fsub(mpm.mp.fdiv(1, G_Partial_tilde[i,i]), 1))
            #print(B_egs[i,i])
        self.H_E[key] = U@B_egs@U.transpose_conj()

        if Return:
            return self.H_E[key]
    
    def DensityMatrixChargeParity(self, debug = False, zero_modes = 'filled'): 
        if self.Gamma_A is None:
            self.Get_Gamma_A(zero_modes=zero_modes)
        gamma_a = MPM_Mat_To_NP(self.Gamma_A[zero_modes])
        Q = np.zeros((2*self.N, 2*self.N), dtype = complex)
        delta = lambda j,k: 1 if j==k  else 0
        exp_am_an = lambda m,n: delta(m,n) + 1j*gamma_a[m, n]
        for i in range(len(Q)):
            for j in range(i):
                Q[i,j] = exp_am_an(i,j)
                Q[j,i] = -exp_am_an(i,j)

        if debug: print(f'Unrounded Q: {pf.pfaffian(Q)}')
        return np.round(pf.pfaffian(Q))*(1j)**(2*self.N)


def Get_Gamma_A_MPM(Ham, debug = False, redo = False, large_Sort_warning = 1000, debug_round = 4, Full_Debug = False):
        A = -2j*Ham

        W, B = mpm.mp.schur(A)
        Gamma_B = mpm.mp.eye(len(B))
        for i in range(len(Gamma_B)):
            if float(mpm.im(B[i,i])<0):
                Gamma_B[i,i]*= -1

        Gamma_A = 1j*W@Gamma_B@W.transpose_conj()

        return Gamma_A
        
def MPM_Mat_To_NP(mat):
    rows = mat.rows
    cols = mat.cols
    return np.array(mat,dtype='complex').reshape((rows,cols)) 


class MajoranaPlotter:      
    def __init__(self, N = 2, Jx = 'All', g = 'All', b_sigma = 'None', pair_sep = 1, site_sep = 1, MajoranaScaleFactors=[0.25, 0.25], MajoranaBoxScaleFactors = [0.25,0.25]):
        self.N = N
        self.Jx = Jx
        self.g = g
        self.b_sigma = b_sigma
        
        if isinstance(self.Jx, str):
            if self.Jx == 'All':
                self.Jx = np.ones(N-1)
            if self.Jx == 'None':
                self.Jx = np.zeros(N-1)
        if isinstance(self.g, str):
            if self.g == 'All':
                self.g = np.ones(N)
            if self.g == 'None':
                self.g = np.zeros(N)
        
        if isinstance(self.b_sigma, str):
            if self.b_sigma == 'None':
                self.b_sigma = np.zeros(N-1)
        
        self.pair_sep = 1
        self.site_sep = 1

        self.SiteLabels = {}
        self.MajoranaScaleFactors = MajoranaScaleFactors
        self.MajoranaBoxScaleFactors = MajoranaBoxScaleFactors
        
        self.MajoranaPairBoxes = {}
        self.Sites = np.arange(N)
        self.Sites_Loc = np.zeros(N)
        for i in self.Sites:
            if i !=0:
                self.Sites_Loc[i] = self.Sites_Loc[i-1]+self.pair_sep+self.site_sep

        
        self.MajoranaMarkers = {}
        self.Cuts = {}
        self.BondLines = {}
        self.ExtraBondLines = {}
        self.max_height = 0
        self.min_height = 0
        
    def BuildBoxForMajoranaPairs(self, color = 'gray', fill = False, kwargs = {}):
        for site in self.Sites:
            site_loc = self.Sites_Loc[site]
            x = site_loc-self.pair_sep*(self.MajoranaScaleFactors[0]+self.MajoranaBoxScaleFactors[0])
            y = 0-self.pair_sep*(self.MajoranaScaleFactors[1]+self.MajoranaBoxScaleFactors[1])

            width = (1+(2*self.MajoranaScaleFactors[0]+self.MajoranaBoxScaleFactors[0]))*self.pair_sep
            height = 2*self.pair_sep*(self.MajoranaScaleFactors[1]+self.MajoranaBoxScaleFactors[1])   
        
            self.MajoranaPairBoxes[site] = plt.Rectangle((x,y), width, height, color = color, fill = fill, **kwargs)

            if self.min_height > y:
                self.min_height = y
            if y+height>self.max_height:
                self.max_height = y+height

    def BuildMajoranasMarkers(self, color1 = 'deeppink', color2 = 'gold', fill1 = True, fill2 = True, kwargs1 = {}, kwargs2 = {}):
        for site in self.Sites:
            site_loc = self.Sites_Loc[site]
            
            x1 = site_loc - self.MajoranaScaleFactors[0]*self.pair_sep
            y1 = 0 - self.MajoranaScaleFactors[1]*self.pair_sep
            
            x2 = x1+self.pair_sep - self.MajoranaScaleFactors[0]*self.pair_sep
            y2 = 0 - self.MajoranaScaleFactors[1]*self.pair_sep

            w = 2*self.MajoranaScaleFactors[0]*self.pair_sep
            h = 2*self.MajoranaScaleFactors[1]*self.pair_sep
            
            marker1 = plt.Rectangle((x1, y1), w, h, color = color1, fill = fill1, **kwargs1)
            marker2 = plt.Rectangle((x2, y2), w, h, color = color2, fill = fill2, **kwargs2)
            
            self.MajoranaMarkers[site] = [marker1, marker2]

            if self.min_height > y1:
                self.min_height = y1
            if y1+h>self.max_height:
                self.max_height = y1+h

    def BuildMajoranaBonds(self, Jx_color = 'black' , g_color = 'deepskyblue', b_sigma_color = 'limegreen', linewidth = 2):
        
        for site in self.Sites:
            site_loc = self.Sites_Loc[site]
            self.BondLines[site] = {}
            if self.g[site]!= 0:
                x1 = site_loc
                y1 = 0
                x2 = x1 + self.pair_sep*(1-self.MajoranaScaleFactors[0])
                y2 = 0
                self.BondLines[site]['g'] = plt.Line2D([x1, x2], [y1,y2], color = g_color, linewidth = linewidth, zorder = -1)
            
            if site < self.N: 
                if self.Jx[site]!= 0:
                    x1 = site_loc+self.pair_sep*(1-self.MajoranaScaleFactors[0])
                    y1 = 0
                    x2 = x1 + self.site_sep*(1+self.MajoranaScaleFactors[0])
                    y2 = 0
                    self.BondLines[site]['Jx'] = plt.Line2D([x1, x2], [y1,y2], color = Jx_color, linewidth = linewidth, zorder = -1)

            if site<self.N-1:
                if self.b_sigma[site]!=0:
                    D = self.pair_sep + self.site_sep
                    x = site_loc+self.pair_sep*(1-self.MajoranaScaleFactors[0])+D/2
                    y = self.pair_sep*self.MajoranaScaleFactors[1]
                    self.BondLines[site]['b_sigma'] = patches.Arc((x,y), width = D, height = D/2, theta1 = 0, theta2 = 180, color = b_sigma_color, linewidth = linewidth)

                    if self.max_height<(y+D/4):
                        self.max_height = (y+D/4)
    
    def AddCut(self, site, color = 'red', linewidth = 2, height_factor = 1, style = '-'):
        x = self.Sites_Loc[site]+self.pair_sep*(1-0.25/2)+0.5*self.site_sep
        top = self.pair_sep*(self.MajoranaScaleFactors[1]+self.MajoranaBoxScaleFactors[1])*height_factor
        bottom = -top

        self.Cuts[site] = plt.Line2D([x,x], [top,bottom], color = color, linewidth = linewidth, linestyle = style)

        if self.max_height<top:
            self.max_height = top
        if self.min_height>bottom:
            self.min_height = bottom

    def AddExtraMajoranaBonds(self, bonds, color = 'blue', linewidth = 2, height_factor = 0.45, style = '-', theta1 = 0, theta2 = 180, updown = 1, height_factor_2 = 0):
        for bond in bonds:
            site1 = bond[0]//2
            site2 = bond[1]//2
            x1 = self.Sites_Loc[site1]+(bond[0]%2)*self.pair_sep*(1-self.MajoranaScaleFactors[0])
            x2 = self.Sites_Loc[site2]+(bond[1]%2)*self.pair_sep*(1-self.MajoranaScaleFactors[0])
            D = np.abs(x2-x1)

            x = D/2+np.min([x1,x2])
            y = self.pair_sep*(self.MajoranaScaleFactors[1])*updown

            Theta1 = (1-updown)*180/2
            Theta2 = (1+updown)*180/2
            height = (self.pair_sep+self.site_sep)*height_factor+height_factor_2*D
            self.ExtraBondLines[str(bond)] = patches.Arc((x,y), width = D, height = height, theta1 = Theta1, theta2 = Theta2, color = color, linewidth = linewidth)

            
            if self.max_height<height*updown:
                self.max_height = height*updown
            if self.min_height >= height*updown:
                self.min_height = height*updown
            


    def Plot(self, dpi = 120):
        fig, ax = plt.subplots(dpi = dpi, layout = 'tight')

        for site in self.Cuts:
            ax.add_line(self.Cuts[site])
               
        for site in self.MajoranaMarkers:
            ax.add_patch(self.MajoranaMarkers[site][0])
            ax.add_patch(self.MajoranaMarkers[site][1])

       
        for site in self.MajoranaPairBoxes:
            ax.add_patch(self.MajoranaPairBoxes[site])
    
        
        for site in self.BondLines:
            for key in self.BondLines[site]:
                if key != 'b_sigma':
                    ax.add_line(self.BondLines[site][key])
                if key == 'b_sigma':
                    ax.add_patch(self.BondLines[site][key])
        
        for bond in self.ExtraBondLines:
            ax.add_patch(self.ExtraBondLines[bond])

        ax.set_xlim([self.Sites_Loc[0]-self.pair_sep, self.Sites_Loc[-1]+2*self.pair_sep])
        ax.set_ylim([self.min_height-0.1, self.max_height+0.1])
        ax.set_aspect('equal')
        
        ax.set_axis_off()
        
        plt.show()
        
        
def Plot_Couplings(ham, diags = [1], antidiags = [], diag_marker = 'x', antidiag_marker = '+', diag_cmap = None, antidiag_cmap = None, figsize = (10,5), dpi = 120, do_abs = False, yscale = 'linear', symlogstart = 10e-6, overlay = {}, title = None):
    if diag_cmap == None:
        diag_cmap = sns.color_palette('crest', as_cmap = True)
    
    if antidiag_cmap == None:
        antidiag_cmap = sns.color_palette('flare', as_cmap = True)
    
    L = len(ham)
    l = L/2
    diags = np.array(diags)
    diags_index_array = np.arange(len(diags))
    antidiags = np.array(antidiags)
    antidiags_index_array = np.arange(len(antidiags))

    if len(diags)>0:
        diag_cmap_for_plot = diag_cmap(diags_index_array/diags_index_array.max())
        if len(diags)>1:
            diag_cmap_cbar_array = np.concatenate([diags_index_array, [diags_index_array.max()+1]])
            diag_norm = colors.BoundaryNorm(diag_cmap_cbar_array-0.5, diag_cmap.N)
            diag_cmap_for_colorbar = cm.ScalarMappable(cmap= diag_cmap, norm = diag_norm)


    if len(antidiags)>0:
        antidiag_cmap_for_plot = antidiag_cmap(antidiags_index_array/antidiags_index_array.max())
        if len(antidiags)>1:
            antidiag_cmap_cbar_array = np.concatenate([antidiags_index_array, [antidiags_index_array.max()+1]])
            antidiag_norm = colors.BoundaryNorm(antidiag_cmap_cbar_array-0.5, antidiag_cmap.N)
            antidiag_cmap_for_colorbar = cm.ScalarMappable(cmap= antidiag_cmap, norm = antidiag_norm)


    fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
    for I in range(len(diags)):
        i = diags[I]
        x = np.arange(0.5*i, L-0.5*i, 1)/L
        couplings = []
        for j in range(L-i):
            coupling = ham[j,j+i]
            if do_abs:
                coupling = np.abs(coupling)
            couplings.append(coupling)

        color = diag_cmap_for_plot[I]
        if len(diags)==1:
            color = 'blue'
        if I ==0:
            ax.plot(x,couplings, diag_marker, color = color , label = 'Diags')    
        ax.plot(x,couplings, diag_marker, color = color)
    
    if len(diags)>1:

        cbar = fig.colorbar(diag_cmap_for_colorbar, label = 'Diagonals')
        cbar.ax.set_yticks(diags_index_array)
        cbar.ax.set_yticklabels(diags)
    plt.legend()

    for I in range(len(antidiags)):
        i = antidiags[I]
        x = np.arange(0.5*np.abs(i), L-0.5*np.abs(i), 1)/L
        couplings = []
        for j in range(L-np.abs(i)):
            if i >=0:
                coupling = ham[j,L-1-j-i]
            if i <0:
                coupling = ham[j-i, L-1-j]
            
            if do_abs:
                coupling = np.abs(coupling)
            
            couplings.append(coupling)

        color = antidiag_cmap_for_plot[I]
        if len(antidiags)==1:
            color = 'red'
        if I ==0:
            ax.plot(x,couplings, antidiag_marker, color = color , label = 'Antidiags')    
        ax.plot(x,couplings, antidiag_marker, color = color)
    
    if len(antidiags)>1:

        anticbar = fig.colorbar(antidiag_cmap_for_colorbar, label = 'Antidiagonals')
        anticbar.ax.set_yticks(antidiags_index_array)
        anticbar.ax.set_yticklabels(antidiags)
    
    for key in overlay:
        x = overlay[key].get('x')
        y = overlay[key].get('y')
        color = overlay[key].get('color', 'black')
        markerfacecolor = overlay[key].get('markerfacecolor', 'none')
        marker = overlay[key].get('marker', 'D')
        markersize = overlay[key].get('markersize', None)        
        plt.plot(x,y,marker, color =color, markerfacecolor=markerfacecolor, label = str(key), ms = markersize)

    
    
    plt.title(title)
    plt.legend()
    if yscale == 'symlog':
        ax.set_yscale(yscale, linthresh = symlogstart)

def str_list_to_list(str_list, TYPE = int):
    return [TYPE(s) for s in str_list[1:-1].split(',')]

def T_Bonds(ham):
    L = len(ham)
    T_Bonds = []
    for i in np.arange(L-1):
        entry = 0
        for j in range(i+1):
            if (i+1+j<L):
                entry += ham[i+1+j, i-j]*(2*j+1)
            
        T_Bonds.append(entry)
    
    T_Bonds = np.array(T_Bonds)
    return T_Bonds

def T_Matrix(ham):
    bonds = T_Bonds(ham)

    if isinstance(ham, np.ndarray):
        mat = np.zeros_like(ham)
    
    if isinstance(ham, mpm.matrix):
        mat = mpm.mp.matrix(len(ham), len(ham))
    for i in range(len(mat)-1):
        bond = bonds[i]
        if isinstance(ham, mpm.matrix):
            bond = mpm.mp.mpc(bond)
        mat[i+1,i] = bond
        mat[i,i+1] = -mat[i+1,i]
    
    return mat

def Get_Diags(ham, diag):
    entries = []
    for i in range(len(ham)-diag):
        entries.append(ham[i+diag, i])
    return entries

def Get_AntiDiags(ham, antidiag):
    entries = []
    for i in range(len(ham)-antidiag):
        entries.append(ham[len(ham)-1-i-antidiag, i])
    return entries

def Fidelity(EH1, EH2):

    if type(EH1) != mpm.matrix or type(EH2) != mpm.matrix:
        print('Matrices need to be mpmath matrices')
        return None
    
    X1, U1 = mpm.mp.eigh(EH1)
    X2, U2 = mpm.mp.eigh(EH2)
    
    B1 = mpm.mp.eye(len(X1))
    B2 = mpm.mp.eye(len(X2))

    for i in range(len(B1)):
        B1[i,i] = 1/(1+mpm.mp.exp(-X1[i]))
    
        B2[i,i] = 1/(1+mpm.mp.exp(-X2[i]))
    
    G1 = U1@B1@U1.transpose_conj()
    X1, U1 = mpm.mp.eigh(G1)

    G2 = U2@B2@U2.transpose_conj()
    X2, U2 = mpm.mp.eigh(G2)

    Id = mpm.mp.eye(len(X1))

    det1 = 1
    det2 = 1
    for i in range(len(X1)):
        det1*= (1-X1[i])/2
        det2*= (1-X2[i])/2


    B1 = mpm.mp.eye(len(X1))
    B2 = mpm.mp.eye(len(X2))
    
    for i in range(len(X1)):
        B1[i,i] = mpm.mp.sqrt((1+X1[i])/(1-X1[i]))
        B2[i,i] = ((1+X2[i])/(1-X2[i]))
    

    det3_mat1 = U1@B1@U1.transpose_conj()  
    det3_mat2 = U2@B2@U2.transpose_conj()

    det3_mat3 = (det3_mat1@det3_mat2@det3_mat1)

    X3, U3 = mpm.mp.eigh(det3_mat3)
    det3 = 1
    for i in range(len(X3)):
        det3*= (1+mpm.mp.sqrt(X3[i]))

    return (det1*det2)**(0.25)*det3**(0.5)

def TimeEvoloveG(G, H, t):
    U = la.expm(-2j*H*t)

    return U.conj().T @ G @ U

def GET_TIME():
    return datetime.now().strftime('%m-%d-%y:%H:%M:%S')

def symlog_cmap(max_val, linthresh, levels = 1024, offset = 0.4, name = 'Code/symlog_cmap', half = None):
    cmap3_table = ascii.read(name+'.csv', delimiter= ';', data_start = 0)

    cmap3_list = []

    for i in range(len(cmap3_table)-1):
        cmap3_list.append(np.array([np.float(cmap3_table['R'][i+1]), np.float(cmap3_table['G'][i+1]), np.float(cmap3_table['B'][i+1]), np.float(cmap3_table['Opacity'][i+1])]))

    cmap3_array = np.array(cmap3_list)[::-1]

    cmap3 = colors.LinearSegmentedColormap.from_list('symlog cmap', cmap3_array, N = len(cmap3_array))
    
   
    #nums = np.tan(np.pi*np.linspace(-0.95, 0.95, levels)/2)
    #nums = nums/nums.max()/2+0.5
    
    next_log = np.ceil(np.log10(max_val))
    last_log = np.floor(np.log10(max_val))
    symlog_length = np.log10(max_val)-np.log10(linthresh)
    length = 2*symlog_length+2+2*(next_log-last_log)


    symlog_levels = int(cmap3.N*symlog_length/length)
    lin_levels = int(cmap3.N*(2+((next_log-last_log)/length))/length)
    

    log_bit = np.linspace(0,1-0*(next_log-last_log)/(length*2), symlog_levels)


    start = offset*log_bit
    end = 1-offset*log_bit[::-1]

    
    mid = np.linspace(start[-1], end[0], lin_levels)

    nums = np.concatenate([start,mid, end])

    #plt.plot(nums)
    
    cmap = cmap3(nums)

    if half == None:
        return colors.LinearSegmentedColormap.from_list('symlog cmap', cmap, N = levels)
    
    if half != None:
        if half == 'lower':
            cmap = cmap3(nums[nums<=0.5])
        if half == 'upper': 
            cmap = cmap3(nums[nums>=0.5])

        return colors.LinearSegmentedColormap.from_list('symlog cmap', cmap, N = levels)


def cmap_trimmer(cmap, left, right=None, levels = 1024, Name = 'Trimmped Cmap'):
    if right == None:
        right = left
    
    if right == 0:
        right = None
    
    if right != None:
        right = -right

    num_arr = np.linspace(0, 1, levels)
    cmap_arr = cmap(num_arr)[left:right]
    
    num_arr2 = np.linspace(0, 1, len(num_arr[left:right]))
    
    cmap_list = []
    for i in range(len(num_arr2)):
        cmap_list.append((num_arr2[i], cmap_arr[i]))
    
    return colors.LinearSegmentedColormap.from_list(Name, cmap_list, N = levels)

def Get_Data(file_names, True_Keywords = [], False_Keywords = [], defect_type = 'Jx', default_defect_x = 'None', debug = False, default_value = 1):
            
    Main_Dicts = []
    for file_name in file_names:

        True_Keywords_TF = np.zeros(len(True_Keywords), dtype=bool)
        False_Keywords_TF = np.zeros(len(False_Keywords), dtype=bool)
        
        for i in range(len(True_Keywords)):
            keyword = True_Keywords[i]
            if keyword in file_name:
                True_Keywords_TF[i] = True
        
        for i in range(len(False_Keywords)):
            keyword = False_Keywords[i]
            if keyword in file_name:
                False_Keywords_TF[i] = True
        
        if len(True_Keywords_TF) == 0:
            True_Keywords_TF = np.array([True])

        if len(False_Keywords_TF) == 0:
            False_Keywords_TF = np.array([False])
        
        if np.all(True_Keywords_TF) and not np.any(False_Keywords_TF):
            with open(file_name, 'rb') as f:
                Main_Dicts.append(pickle.load(f))
    
    N_Coords = []
    defect_Coords = []
    defect_x_Coords = []
    for i in range(len(Main_Dicts)):
        dat = Main_Dicts[i]

        N = dat['N']

        defect = 1
        defect_x = default_defect_x
        if isinstance(dat[defect_type], (list, np.ndarray)):
            if isinstance(dat[defect_type], (list)):
                dat[defect_type] = np.array(dat[defect_type])

            if not np.all(dat[defect_type] == default_value):
                defect_locs = np.where(dat[defect_type]!=default_value)[0]
                 
                defects = []
                defect_xs = []
                for defect_loc in defect_locs:
                    defects.append(dat[defect_type][defect_loc])
                    defect_x = fractions.Fraction(defect_loc+1,N)
                    if defect_x.denominator == 1:
                        defect_x = f'{defect_x.numerator}'
                    else:
                        defect_x = f'{defect_x.numerator}/{defect_x.denominator}'
                    defect_xs.append(defect_x)
                

                if len(defect_xs) == 1:
                    defect_xs = defect_xs[0]
                if len(defects) == 1:
                    defects = defects[0]
                
                if type(defect_xs) == list:
                    if len(defect_xs) == len(dat[defect_type]):
                        defect_xs = 'All Bonds'

                    if len(defects) == len(dat[defect_type]):
                        defects = defects[0]
                    
                
            if np.all(dat[defect_type] == default_value):
                defects = 'None'
                defect_xs = 'None'


        if debug: print(f'N: {N}, Defect: {defects}, defect_loc: {defect_locs}, defect_xs: {defect_xs}')
        
        if N not in N_Coords:
            N_Coords.append(N)
        
        if str(defects) not in defect_Coords:
            defect_Coords.append(str(defects))

        if str(defect_xs) not in defect_x_Coords:
            if defect_xs == 'None':
                defect_x_Coords.insert(0, str(defect_xs))
            else:
                defect_x_Coords.append(str(defect_xs))
        
    #N_Coords = np.array(N_Coords)
    #N_Coords.sort()
    #defect_Coords = np.array(defect_Coords)
    #defect_Coords.sort()


    coords = {'N':N_Coords, 'Defect':defect_Coords, 'Defect x':defect_x_Coords}
    if debug:
        print(coords)
    Main_DA = xr.DataArray(coords=coords, dims = list(coords.keys()))
    Main_DA = Main_DA.astype(object)


    for i in range(len(Main_Dicts)):
        dat = Main_Dicts[i]
        N = dat['N']
        defect = 1
        defect_x = default_defect_x
        dps = dat['dps']
        if isinstance(dat[defect_type], (list, np.ndarray)):
            if isinstance(dat[defect_type], (list)):
                dat[defect_type] = np.array(dat[defect_type])

            if not np.all(dat[defect_type] == default_value):
                defect_locs = np.where(dat[defect_type]!=default_value)[0]
                
                defects = []
                defect_xs = []
                for defect_loc in defect_locs:
                    defects.append(dat[defect_type][defect_loc])
                    defect_x = fractions.Fraction(defect_loc+1,N)
                    if defect_x.denominator == 1:
                        defect_x = f'{defect_x.numerator}'
                    else:
                        defect_x = f'{defect_x.numerator}/{defect_x.denominator}'
                    defect_xs.append(defect_x)
                
                if len(defect_xs) == 1:
                    defect_xs = defect_xs[0]
                if len(defects) == 1:
                    defects = defects[0]

                if type(defect_xs) == list:
                    if len(defect_xs) == len(dat[defect_type]):
                        defect_xs = 'All Bonds'

                    if len(defects) == len(dat[defect_type]):
                        defects = defects[0]
                
            if np.all(dat[defect_type] == default_value):
                defects = 'None'
                defect_xs = 'None'
            
            defects = str(defects)
            defect_xs = str(defect_xs)
        if type(Main_DA.loc[N, defects, defect_xs].item()) == dict:
            if Main_DA.loc[N, defects, defect_xs].item()['dps'] < dps:
                Main_DA.loc[N, defects, defect_xs] = dat
        if type(Main_DA.loc[N, defects, defect_xs].item()) != dict:
            Main_DA.loc[N, defects, defect_xs] = dat

    for index in Main_DA.coords.to_index():
        if type(Main_DA.loc[index].item()) == dict:
            
            N = Main_DA.loc[index].item()['N']

            temp_Jx = Main_DA.loc[index].item()['Jx']
            temp_Jy = Main_DA.loc[index].item()['Jy']
            temp_g = Main_DA.loc[index].item()['g']
            temp_b_sigma = Main_DA.loc[index].item()['b_sigma']
             
            bc = Main_DA.loc[index].item()['boundaries']
            Main_DA.loc[index].item()['Model'] = MajoranaFermionChain(N=N, Jx = temp_Jx, Jy = temp_Jy, g = temp_g, b_sigma = temp_b_sigma, boundaries = bc)

    return Main_DA

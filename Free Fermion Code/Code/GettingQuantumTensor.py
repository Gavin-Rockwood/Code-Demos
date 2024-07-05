import numpy as np

import xarray as xr

import scipy.linalg as la

import itertools as itert

from astropy.io import ascii

import sys
import os

import glob

import pickle

import MajoranaFermionChain as MFC


def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def Get_T(model, Round = 14):
    T = np.empty((model.N, model.N), dtype=complex)

    Gamma = model.Get_Gamma_A(Return = True)
    G = (np.eye(len(Gamma)) + 1j*Gamma)/2

    delta_ij = lambda i,j: 1 if i==j else 0

    for i in range(model.N):
        for j in range(model.N):
            T[i,j] = 2*(2*G[2*i+1, 2*j+1] - 2j*G[2*i, 2*j+1] - delta_ij(i,j)) # This delta is missing the factor of 1/2 out front. This may have something to do with being in the majorana basis for this calculation. Hmm. 

    return T


def Get_F(T, Tp):
    return np.sqrt(np.abs(la.det((T+Tp)/2)))

def Get_g(Model_DA, debug  = False, Return_K_DA=False):
    
    N = Model_DA.attrs['N']
    
    coords = xr.Coordinates({'row':np.arange(N), 'col':np.arange(N)})

    K_DA = xr.DataArray(None, coords=Model_DA.coords.merge(coords).coords, dims = Model_DA.dims+('row', 'col')).astype(complex)

    if debug:print(f'K_DA Coords:\n{K_DA.coords}')

    it = np.nditer(Model_DA.to_numpy(), flags = ['multi_index', 'refs_ok'])
    for x in it:
        model = Model_DA[it.multi_index]
        
        T, Z = la.schur(Get_T(model.item()))
        
        T = np.round(T, 14)

        T[np.imag(T)==0] = np.real(T[np.imag(T)==0])

        K = Z@np.diag(np.log(np.diagonal(T)))@np.conj(Z.T) #MFC.Matrix_Log(Get_T(model.item()))
        
        K_DA.loc[model.coords] = K.copy()
    
    coords = xr.Coordinates({'mu': list(Model_DA.dims), 'nu':list(Model_DA.dims)})
    g_DA = xr.DataArray(None, coords = Model_DA.coords.merge(coords).coords, dims = Model_DA.dims+('mu', 'nu')).astype(float)

    #print(f'g_DA.coords: {g_DA.coords}')
    Derivative_Dict = {}
    for dim in Model_DA.dims:
        #print(dim)
        Derivative_Dict[dim] = K_DA.differentiate(dim)



    it = np.nditer(Model_DA.to_numpy(), flags = ['multi_index', 'refs_ok'])
    for x in it:
        #print(f'x: {Model_DA[it.multi_index].coords}\n')
        model = Model_DA[it.multi_index]
        for mu in Derivative_Dict.keys():
            for nu in Derivative_Dict.keys():
                pos = model.coords
                div_mu = Derivative_Dict[mu].loc[pos].to_numpy()
                div_nu = Derivative_Dict[nu].loc[pos].to_numpy()
                div_nu = np.conj(div_nu).T
                g_DA.loc[pos].loc[(mu, nu)] = np.trace(div_mu@div_nu)/8 # I added the (-1) to the front to keep the metric positive instead of negative 


    if Return_K_DA:
        return g_DA, K_DA
    return g_DA



def Christoffel_Elements(g, pdg, i_mu, i_nu, i_rho, g_inv = None): # Gamma_{mu, nu}^rho = 0.5 g^{rho, sigma}(pd_mu g_{sigma, nu} + pd_nu g_{sigma mu} - pd_sigma g_{mu nu})

    if g_inv is None:
        g_inv = la.inv(g)

    term1 = 0
    for i_sigma in range(len(g)):
        term1 += g_inv[i_rho, i_sigma]*pdg[i_mu, i_sigma, i_nu]
    
    term2 = 0
    for i_sigma in range(len(g)):
        term2 += g_inv[i_rho, i_sigma]*pdg[i_nu, i_sigma, i_mu]
    

    term3 = 0
    for i_sigma in range(len(g)):
        term3 += g_inv[i_rho, i_sigma]*pdg[i_sigma, i_mu, i_nu]
    
    return 0.5*(term1 + term2 - term3)
    

def Single_Site_Christoffel(g, pdg): # Gamma_{mu, nu}^rho = 0.5 g^{rho, sigma}(pd_mu g_{sigma, nu} + pd_nu g_{sigma mu} - pd_sigma g_{mu nu})
    n = len(g)
    g_inv = la.inv(g)
    
    Gamma = np.zeros((n,n,n)) # mu, nu, rho

    for i_mu in range(n):
        for i_nu in range(n):
            for i_rho in range(n):
                Gamma[i_mu, i_nu, i_rho] = Christoffel_Elements(g, pdg, i_mu, i_nu, i_rho, g_inv = g_inv)

    return Gamma

def Get_Christoffel_Connection(g_DA, dims):
    coords = list(g_DA.dims)[0:dims]

    div_dict = {}
    for coord in coords: # deriv index then two more
        div_dict[coord] = g_DA.differentiate(coord)
    

    dict_of_coords = {}
    for coord in coords:
        dict_of_coords[coord] = g_DA.coords[coord].to_numpy()
     
    Coords_For_DA = xr.Coordinates({**dict_of_coords, **{'mu':coords, 'nu':coords, 'rho':coords}})

    Christoffel_DA = xr.DataArray(None, coords = Coords_For_DA, dims = coords+['mu', 'nu', 'rho']).astype(complex)


    it = itert.product(*list(dict_of_coords.values())) 
    for x in it:
        g = g_DA.loc[x].to_numpy()
        pdg = np.zeros((len(g), len(g), len(g)), dtype=complex)

        for i_mu in range(len(g)):
            for i_nu in range(len(g)):
                for i_rho in range(len(g)):
                    #print(div_dict[coords[i_rho]].loc[x][i_mu, i_nu].item())
                    pdg[i_rho, i_mu, i_nu] = div_dict[coords[i_rho]].loc[x][i_mu, i_nu].item()
        
        Christoffel_DA.loc[x] = Single_Site_Christoffel(g, pdg)
    
    return Christoffel_DA


def Get_Ricci_Tensor(g_DA, Christoffel_DA, dims):
    coords = list(g_DA.dims)[0:dims]

    div_dict = {}
    for coord in coords: # deriv index then two more
        div_dict[coord] = Christoffel_DA.differentiate(coord)

    dict_of_coords = {}
    for coord in coords:
        dict_of_coords[coord] = g_DA.coords[coord].to_numpy()
     
    Coords_For_DA = xr.Coordinates(dict_of_coords)

    Ricci_DA = xr.DataArray(None, coords = g_DA.coords, dims = g_DA.dims).astype(float)

    it = itert.product(*list(dict_of_coords.values())) 
    for x in it:
        g = g_DA.loc[x].to_numpy()
        Christoffel = Christoffel_DA.loc[x].to_numpy()

        
        for i_mu in range(len(g)):
            for i_nu in range(len(g)):
                term1 = 0
                term2 = 0
                term3 = 0
                for i_rho in range(len(g)):
                    term1 += div_dict[coords[i_rho]].loc[x][i_mu, i_nu, i_rho].item() - div_dict[coords[i_nu]].loc[x][i_mu, i_rho, i_rho].item()
                    for i_sigma in range(len(g)):
                        term2 += Christoffel[i_mu, i_nu, i_sigma]*Christoffel[i_sigma, i_rho, i_rho] 
                        term3 += Christoffel[i_mu, i_rho, i_sigma]*Christoffel[i_sigma, i_nu, i_rho]
            
                Ricci_DA.loc[x][i_mu, i_nu] = term1 + (term2 - term3) 

    return Ricci_DA

def Get_Ricci_Scalar(g_DA, Ricci_DA, dims):
    coords = list(g_DA.dims)[0:dims]
    
    dict_of_coords = {}
    for coord in coords:
        dict_of_coords[coord] = g_DA.coords[coord].to_numpy()
     
    Coords_For_DA = xr.Coordinates(dict_of_coords)
    Ricci_Scalar_DA = xr.DataArray(None, coords = Coords_For_DA, dims = coords).astype(float)

    it = itert.product(*list(dict_of_coords.values())) 
    for x in it:
        g = g_DA.loc[x].to_numpy()
        g_inv = la.inv(g)
        
        ricci = Ricci_DA.loc[x].to_numpy()
        
        Ricci_Scalar_DA.loc[x] = np.trace(g_inv@ricci) 

    return Ricci_Scalar_DA


def Build_Config_Space_Patch_Files_h_gamma(h_range, gamma_range, step_size, Save_Name, buffer_size = 20, patch_number = 100, N = 16, Patch_Path = 'Patch_Files/', boundaries = 'periodic'):
    if type(h_range) == str:
        h_range = MFC.str_list_to_list(h_range, TYPE = float)
    if type(gamma_range) == str:
        gamma_range = MFC.str_list_to_list(gamma_range, TYPE = float)
    
    h_array = np.round(np.arange(h_range[0], h_range[1]+step_size, step_size), 14)
    gamma_array = np.round(np.arange(gamma_range[0], gamma_range[1]+step_size, step_size), 14)

    patches = []

    h_patch_size = int(np.ceil(len(h_array)/patch_number))
    gamma_patch_size = int(np.ceil(len(gamma_array)/patch_number))

    print(f"Total Number of Points: {len(h_array)*len(gamma_array)}\n   h Patch Size: {h_patch_size}\n  gamma_patch_size: {gamma_patch_size}")
    for i in range(patch_number):
        for j in range(patch_number):
            h_start_buffer = 1
            if i == 0:
                h_start_buffer = 0
            
            h_end_buffer = 1
            if i == patch_number-1:
                h_end_buffer = 0

            gamma_start_buffer = 1
            if j== 0:
                gamma_start_buffer = 0
            
            gamma_end_buffer = 1
            if j == patch_number-1:
                gamma_end_buffer = 0


            h_start = h_patch_size*i-buffer_size*h_start_buffer 
            h_end = h_patch_size*(i+1)+buffer_size*h_end_buffer
            if h_end >= len(h_array):
                h_end = len(h_array)
            
            gamma_start = gamma_patch_size*j-buffer_size*gamma_start_buffer 
            gamma_end = gamma_patch_size*(j+1)+buffer_size*gamma_end_buffer
            
            if gamma_end >= len(gamma_array):
                gamma_end = len(gamma_array)

            patch_dict = {'h_array':h_array[h_start:h_end], 'gamma_array':gamma_array[gamma_start:gamma_end], 'buffers':{'h_start': h_start_buffer*buffer_size, 'h_end': h_end_buffer*buffer_size, 'gamma_start': gamma_start_buffer*buffer_size, 'gamma_end': gamma_end_buffer*buffer_size}, 'patch_ID':[i,j]}

            patches.append(patch_dict)
    for patch in patches:
        print(f"Doing Patch: {patch['patch_ID']} out of [{patch_number-1}, {patch_number-1}]:")
        Model_DA = xr.DataArray(None, coords=(dict(h = patch['h_array'], gamma=patch['gamma_array'])), dims = ('h', 'gamma')).astype(object)

        Model_DA.attrs['N'] = N
        Model_DA.attrs['Buffers'] = patch['buffers']
        Model_DA.attrs['Patch_ID'] = patch['patch_ID']
        Model_DA.attrs['Patch_Number'] = patch_number
        Model_DA.attrs['Step_Size'] = step_size
        Model_DA.attrs['N'] = N
        
        print(f"    Building Models\n       Number of Models in Patch: {len(patch['gamma_array'])*len(patch['h_array'])}")
        for gamma in patch['gamma_array']:
            for h in patch['h_array']:
                Model_DA.loc[(h, gamma)] = MFC.MajoranaFermionChain(N = N, Jx = (1+gamma)/2, Jy = (1-gamma)/2, g = h, cutoff = 13)
                
                Model_DA.loc[(h, gamma)].item().Ham_Builder()
                B, W = la.schur(np.real(2j*Model_DA.loc[(h, gamma)].item().Ham))

                
                if np.abs(np.round(la.det(W),10)) != 1:
                    gamma_prime = gamma + (gamma_array[1]-gamma_array[0])*1e-7
                    Model_DA.loc[(h, gamma)] = MFC.MajoranaFermionChain(N = N, Jx = (1+gamma_prime)/2, Jy = (1-gamma_prime)/2, g = h, cutoff = 13)
                    Model_DA.loc[(h, gamma)].item().Ham_Builder()
                    B, W = la.schur(np.real(2j*Model_DA.loc[(h, gamma)].item().Ham))
                    Model_DA.loc[(h, gamma)].item().Get_Gamma_A(Return = False, Given_Schur = [B,W])

                else:
                    Model_DA.loc[(h, gamma)].item().Get_Gamma_A(Return = False, Given_Schur = [B,W])
        print(f"        Model_DA.coords: {Model_DA.coords}")
        print(f'    Getting g')
        g_DA = Get_g(Model_DA)
        g_DA.attrs = Model_DA.attrs
        
        print(f'    Getting Christoffel')
        Christoffel_Connection_DA = Get_Christoffel_Connection(g_DA, 2)
        
        print(f'    Getting Ricci Tensor')
        Ricci_Tensor_DA = Get_Ricci_Tensor(g_DA, Christoffel_Connection_DA, 2)
        

        print(f'    Getting Ricci Scalar')
        Ricci_Scalar_DA = Get_Ricci_Scalar(g_DA, Ricci_Tensor_DA, 2)        
        Ricci_Scalar_DA.attrs = Model_DA.attrs
        
        file_name_g = Patch_Path+Save_Name+'_g_'+str(patch['patch_ID'])
        file_name_ricci_scalar = Patch_Path+Save_Name+'_Ricci_Scalar_'+str(patch['patch_ID'])

        pickle.dump(g_DA, open(file_name_g+'.p', 'wb'))
        pickle.dump(Ricci_Scalar_DA, open(file_name_ricci_scalar+'.p', 'wb'))
        print(f'-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')


def Combine_Batch_h_gamma(Batch_Name, Patch_Path = 'QuantumTensorFiles/Patch_Files/', buffer = None, patch_number = None):#, Save_Path = 'Full_Configuration_Space_Files/'):
    files = {}
    files['Metric'] = glob.glob(Patch_Path+Batch_Name+'_g_*.p')
    files['Ricci Scalar'] = glob.glob(Patch_Path+Batch_Name+'_Ricci_Scalar_*.p') 
    DA_List = []
    for bundle in files:
        for data_file in files[bundle]:
            with open(data_file,'rb') as file:
                temp_DA = pickle.load(file)
                temp_DA.name = bundle
            #print(temp_DA.coords)
            buffers = None
            if 'Buffers' in temp_DA.attrs:
                buffers = temp_DA.attrs['Buffers']
            
            if buffers == None:
                if buffer == None:
                    print(f'Please Provide Buffers.')
                    return None

                buffers = {'h_start':buffer, 'h_end':buffer, 'gamma_start':buffer, 'gamma_end':buffer}

            if 'Patch_ID' in temp_DA.attrs:
                Patch_ID = temp_DA.attrs['Patch_ID']

            else:
                Patch_ID = MFC.str_list_to_list('['+find_between_r(data_file, '[', ']')+']')

            if 'Patch_Number' in temp_DA.attrs:
                patch_number = temp_DA.attrs['Patch_Number']
            
            if patch_number is None:
                print(f'Provide a patch number')
                return None

            i = Patch_ID[0]
            j = Patch_ID[1]

            h_start_buffer = 1
            if i == 0:
                h_start_buffer = 0
            
            h_end_buffer = 1
            if i == patch_number-1:
                h_end_buffer = 0

            gamma_start_buffer = 1
            if j== 0:
                gamma_start_buffer = 0
            
            gamma_end_buffer = 1
            if j == patch_number-1:
                gamma_end_buffer = 0
            
            #print(temp_DA.coords)
            h_start = (buffers['h_start'])*h_start_buffer
            h_end = -(buffers['h_end'])*h_end_buffer+(1-h_end_buffer)*(len(temp_DA.coords['h'].values))
            gamma_start = (buffers['gamma_start'])*gamma_start_buffer
            gamma_end = -(buffers['gamma_end'])*gamma_end_buffer+(1-gamma_end_buffer)*(len(temp_DA.coords['gamma'].values))
            
            #print(f"Patch ID: {Patch_ID}")
            #print([h_start,h_end, gamma_start,gamma_end])

            temp_DA = temp_DA[h_start:h_end, gamma_start:gamma_end]
            #print(temp_DA.attrs)
            #print(temp_DA.coords)
            DA_List.append(temp_DA)
    
    
    return xr.combine_by_coords(DA_List, combine_attrs='drop_conflicts')
    

def Full_h_gamma_Run(h_range, gamma_range, step_size, Run_Name, buffer_size = 20, patch_number = 100, N = 16, Patch_Path = 'QuantumTensorFiles/Patch_Files/', Save_Path = 'Full_Configuration_Space_Files/'):
    CHECK_FOLDER = os.path.isdir(Patch_Path)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(Patch_Path)
        print("created folder : ", Patch_Path)
    else:
        print(Patch_Path, "folder already exists.")

    CHECK_FOLDER = os.path.isdir(Save_Path)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(Save_Path)
        print("created folder : ", Save_Path)
    else:
        print(Save_Path, "folder already exists.")
    
    
    
    Build_Config_Space_Patch_Files_h_gamma(h_range=h_range, gamma_range=gamma_range, step_size=step_size, Save_Name=Run_Name, buffer_size=buffer_size, patch_number=patch_number, N = N, Patch_Path = Patch_Path)    
    final = Combine_Batch_h_gamma(Batch_Name = Run_Name, Patch_Path = Patch_Path)
    
    with open(Save_Path+Run_Name+'.p', 'wb') as file:
        pickle.dump(final, file)

if __name__ == '__main__':
    '''To Run this, the only argument it takes is the path to the init file. For example: QuantumTensorFiles/init_files/StripOfParameterSpace_Periodic.csv'''
    params = ascii.read(sys.argv[1], delimiter = ';')
    params = params.to_pandas().to_dict(orient = 'records')[0]

    i = 1
    name = params['Run_Name']+'_N_'+str(params['N'])+'_Step_Size_'+str(params['step_size'])
    params['Run_Name'] = name
    while True:
        if len(glob.glob(params['Patch_Path']+name+'_*')) != 0:
            name =  params['Run_Name']+'_('+str(i)+')'
        else:
            params['Run_Name'] = name
            break

    Full_h_gamma_Run(**params)


# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('../')
from deepspt_src import *

"""Generate a simulated data """

# controls
plot = False
save = True

# variables
n_per_clean_diff = 10 # 1000 per n_classes
n_classes = 4 # diffusion types if 5 stuck is added
n_changing_traces = 4 * n_classes * n_per_clean_diff # 80% data is heterogeneous

print(n_per_clean_diff, n_changing_traces, n_per_clean_diff*n_classes+n_changing_traces)

# %%
random_D = True
multiple_dt = False # not implemented but as D is multiplied with dt for varying D is the same

Nrange = [5,600]
Brange = [0.05,1] 
Rrange = [5,25] 
subalpharange = [0,0.7]
superalpharange = [1.3, 2] 
Qrange = [1,16] 
Drandomrange = [10**-4, 0.5]
Dfixed = 0.1
dir_motion = 'active'

dim = 2 # 2D or 3D
dt = 1 # s
max_changepoints = 4 # number of times changing diffusion traces can change
min_parent_len = 5 # minimum length of subtrace
total_parents_len = Nrange[1] # max len of changing diff traces
path = 'tracks/'
output_name = 'LargerConfinement_SimDiff_coloc_dim'+str(dim)

if random_D:
    Dname = 'Drandom{}-{}'.format(Drandomrange[0],Drandomrange[1])
else:
    Dname = 'Dfixed{}'.format(Dfixed)

note = Dname +'_dim{}'.format(dim)+'_dt{:.1e}'.format(dt)+\
       '_N{}-{}'.format(Nrange[0],Nrange[1])+\
       '_B{}-{}'.format(Brange[0],Brange[1])+\
       '_R{}-{}'.format(Rrange[0],Rrange[1])+\
       '_subA{}-{}'.format(subalpharange[0],subalpharange[1])+\
       '_superA{}-{}'.format(superalpharange[0],superalpharange[1])+\
       '_Q{}-{}'.format(Qrange[0],Qrange[1])+\
       '_DMtype_{}'.format(dir_motion)

print(n_per_clean_diff, n_changing_traces)
print('note as:', note)

# Clean diffusion types
print("Generating data")
params_matrix = Get_params(n_per_clean_diff, dt, random_D, multiple_dt,
                           Nrange = Nrange, Brange = Brange, 
                           Rrange = Rrange, 
                           subalpharange = subalpharange,
                           superalpharange = superalpharange, 
                           Qrange = Qrange, 
                           Drandomrange = Drandomrange,
                           Dfixed = Dfixed)
NsND, NsAD, NsCD, NsDM, NstD = [params_matrix[i] for i in range(5)]
Ds, r_cs, ellipse_dims, angles, vs, wiggle, r_stuck, subalphas, superalphas, sigmaND, sigmaAD, sigmaCD, sigmaDM, sigmaStD = params_matrix[7:]


normal_diff = Gen_normal_diff(Ds, dt, sigmaND, NsND, dim=dim, min_len=min_parent_len)
print("\tnormal done")
if dir_motion=='active':
    directed_diff = Gen_directed_diff(Ds, dt, vs, sigmaDM, NsDM, dim=dim, min_len=min_parent_len)
    print("\tdirected done")
elif dir_motion=='super':
    directed_diff = Gen_anomalous_diff(Ds, dt, superalphas, sigmaDM, NsDM, dim=dim, min_len=min_parent_len)
    print("\tdirected done")

confined_diff = Gen_new_confined_diff(Ds, dt, sigmaCD, NsCD, dim=dim, min_len=min_parent_len, ellipse_dims=ellipse_dims, angles=angles)
print("\tconfined done")
subnormal_diff = Gen_anomalous_diff(Ds, dt, subalphas, sigmaAD, NsAD, dim=dim, min_len=min_parent_len)
print("\tsubdiffusion done")
#stuck_diff = Gen_stuck_diff(wiggle, dt, r_stuck, sigmaStD, NstD, dim=dim)
#print("\tstuck done")

#print("\tsuperdiffusion done")
clean_diffusion_list = [] + normal_diff + directed_diff + confined_diff + subnormal_diff #+ supernormal_diff
print(len(clean_diffusion_list))
# 0 is normal diff
# 1 is directed motion
# 2 is confined diffusion
# 3 is anomalous diffusion sub
# 4 is anomalous diffusion super
clean_labels = [0] * n_per_clean_diff + [1] * n_per_clean_diff + [2] * n_per_clean_diff + [3] * n_per_clean_diff #+ [4] * n_per_clean_diff
print(len(clean_labels))

clean_time_resolved_labels = [[0]*len(t) for t in normal_diff] +\
                       [[1]*len(t) for t in directed_diff] +\
                       [[2]*len(t) for t in confined_diff] +\
                       [[3]*len(t) for t in subnormal_diff]#+\
                       #[[4]*len(t) for t in supernormal_diff]

# Changing diffusion types
s = datetime.datetime.now()
changing_diffusion_list, changing_label_list = Gen_changing_diff(n_changing_traces, 
                                                                 max_changepoints, 
                                                                 min_parent_len, 
                                                                 total_parents_len, 
                                                                 dt, random_D=random_D, 
                                                                 n_classes=n_classes, dim=dim,
                                                                 Nrange = Nrange, Brange = Brange, 
                                                                 Rrange = Rrange, 
                                                                 subalpharange = subalpharange,
                                                                 superalpharange = superalpharange, 
                                                                 Qrange = Qrange, 
                                                                 Drandomrange = Drandomrange,
                                                                 Dfixed = Dfixed,
                                                                 DMtype=dir_motion)
print("\tchanging diffusion done", datetime.datetime.now()-s)

if plot:
    if dim == 2:
        for i, cc in enumerate(clean_diffusion_list):
            savename_clean = 'tracks/figures/'+timestamper()+'_simulated_clean_diff'
            plot_diffusion(cc, clean_time_resolved_labels[i])#, savename=savename_clean)
        for i, cd in enumerate(changing_diffusion_list):
            savename = 'tracks/figures/'+timestamper()+'_simulated_diff'
            plot_diffusion(cd, changing_label_list[i])#, savename=savename)
    if dim == 3:
        plot_3Ddiffusion(clean_diffusion_list, clean_time_resolved_labels)
        plot_3Ddiffusion(changing_diffusion_list, changing_label_list)

if save:
    output = clean_diffusion_list + changing_diffusion_list
    output_labels = clean_time_resolved_labels + changing_label_list

    date = str(datetime.datetime.now().year)+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)
    T = str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
    date_T = date+T

    print('save as:', path+date_T+'_'+output_name+'_ntraces'+str(len(output))+'_'+note+"_X.pkl")

    pickle.dump(output, open(path+date_T+'_'+output_name+'_ntraces'+str(len(output))+'_'+note+"_X.pkl", "wb"))
    pickle.dump(output_labels, open(path+date_T+'_'+output_name+'_'+note+"_timeresolved_y.pkl", "wb"))

    #pickle.dump(clean_diffusion_list, open(path+date_T+'_simulated_clean_dim'+str(dim)+'_'+note+'_X.pkl', "wb"))
    #pickle.dump(clean_labels, open(path+date_T+'_simulated_clean_dim'+str(dim)+'_'+note+'_y.pkl', "wb"))


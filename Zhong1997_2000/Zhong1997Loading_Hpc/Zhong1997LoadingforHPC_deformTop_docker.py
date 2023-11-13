#!/usr/bin/env python
# coding: utf-8

# In[1]:


import underworld as uw
import underworld.function as fn
from underworld import UWGeodynamics as GEO
import numpy as np
import math

from mpi4py import MPI as _MPI
comm = _MPI.COMM_WORLD
rank = comm.rank
size = comm.size


# In[2]:


u = GEO.UnitRegistry

# solver parameters
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-2
GEO.rcParams['initial.nonlinear.max.iterations'] = 50
GEO.rcParams["nonlinear.tolerance"] = 1e-2
GEO.rcParams['nonlinear.max.iterations'] = 50
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 30
GEO.rcParams["swarm.particles.per.cell.2D"] = 30

GEO.rcParams["surface.pressure.normalization"] = True
GEO.rcParams["pressure.smoothing"] = True


# In[3]:


# input parameters
type_bcs = "_FreeSlipBot"
#type_bcs = "_LithoBot"
fdir_output = "DefrmTop_Loading"+ type_bcs+"_resy480"

xmin_box,xmax_box = -600,600             # unit [km]
ymin_box,ymax_box = -600,0 # unit [km]

x_box = xmax_box-xmin_box
y_box = -ymin_box+ymax_box

x_res,y_res = 960,480
npoints = 1200
dx = x_box/x_res
dy = y_box/y_res
    
conv_vel = 1.0 * u.centimeter / u.year

# checkpoint_interval = 0.1 * u.megayears
# dt = 2.5 * u.kiloyear
# Total_Convergence  = 500 *u.kilometer
# Total_Time = (Total_Convergence / conv_vel).to(u.megayear)


# scaling
ref_velocity = 1. * u.centimeter / u.year
ref_density = 3300. * u.kilogram / u.meter**3
ref_length = 600. * u.kilometer  
ref_gravity =  10.0 * u.meter / u.second**2
gravity = 10.0 * u.meter / u.second**2
#ref_vicosity = 1e21 * u.pascal * u.second

T0 = 273.15 * u.degK  # 0 * u.degC
Tz = 1573.15 * u.degK # 1300 * u.degC at litho bottom
# Tz = 1300 + 273.15 + (-ymin_box-ml_thickness)*120

bodyforce = ref_density * ref_gravity
KL = ref_length
Kt = KL / ref_velocity
KM = bodyforce * KL**2 * Kt**2
KT = (Tz - T0)

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
GEO.scaling_coefficients["[temperature]"] = KT

if uw.mpi.rank == 0:
    print('Length, km = ', GEO.dimensionalise(1., u.kilometer))
    print('Time, Myr = ',GEO.dimensionalise(1., u.megayear))
    print('Pressure, MPa = ',GEO.dimensionalise(1., u.megapascal))
    print('Temperature, K = ',GEO.dimensionalise(1., u.degK))
    print('Velocity, cm/yr = ',GEO.dimensionalise(1., u.centimeter / u.year))
    print('Viscosity, Pa S = ',GEO.dimensionalise(1.,u.pascal * u.second))
    
    
dx_nd = GEO.nd(dx*u.kilometer)
dy_nd = GEO.nd(dy*u.kilometer)


# In[4]:


Model = GEO.Model(elementRes=(x_res, y_res),
                  minCoord=(xmin_box*u.kilometer, ymin_box*u.kilometer),
                  maxCoord=(xmax_box*u.kilometer, ymax_box*u.kilometer),
                  gravity=(0.0, -gravity))
Model.outputDir=fdir_output
Model.minStrainRate = 1e-18 / u.second


# In[5]:


# from underworld import visualisation as vis
# Fig = vis.Figure(resolution=(1200,600),rulers=True,margin = 60)
# Fig.Mesh(Model.mesh)
# Fig.show()


# In[6]:


# kkt = 250 
# top = Model.top_wall
# nox = len(top) 
# topo_test = np.zeros(nox)
# v_or_t,all0,ramp0,boxl = 5., 40, 20., 600. 
# pi = np.pi
# d = KL.m
# xcoordi =  Model.mesh.data[top][:,0]
# topo_s0 = np.zeros(nox)
# acosine = np.zeros((nox, kkt))
# ff = np.zeros(kkt)
# for k in range(1, kkt+1):
#     wavelength1 = 2 * boxl / k/d
#     aki = np.float128(2 * pi / wavelength1)
    
#     for ij in range(nox):
#         acosine[ij, k-1] = np.cos(xcoordi[ij] * aki)

#     ak = boxl / (pi * k)
#     ff[k-1] = 2 * v_or_t * ak * ak / (ramp0 * boxl) * (np.cos(all0/ak) - np.cos((ramp0 + all0)/ak))

#     for ij in range(nox):
#         topo_s0[ij] += ff[k-1] * acosine[ij, k-1]/d


# In[7]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# fig, ax1 = plt.subplots(nrows=1, figsize=(8,3))
# ax1.set(xlabel='Distacne [km]', ylabel='Topography [km]') 
# #ax1.set_xlim([0,1500])
# ax1.plot(xcoordi*d,topo_s0*d,c="k",label="Vel")


# In[8]:


# from scipy.interpolate import interp1d

# x = Model.mesh.data[Model.top_wall][:, 0]
# f = interp1d(GEO.nd(xcoordi_double*u.kilometer), GEO.nd(topo_s0_double*u.kilometer), kind='cubic', fill_value='extrapolate')
# y = f(x) 


# In[9]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# fig, ax1 = plt.subplots(nrows=1, figsize=(8,3))
# ax1.set(xlabel='Distacne [km]', ylabel='Topography [km]') 
# #ax1.set_xlim([0,1500])
# ax1.plot(x,y,c="k",label="Vel")
# ax1.plot(x, GEO.nd(topo_s0_double*u.kilometer),c="r",label="Vel")


# In[10]:


# axis_moho = condition 
# IndexSet = Model.mesh.specialSets["Empty"]
# for index in axis_moho:
#     IndexSet.add(index)


# In[11]:


D_moho = -35.*u.kilometer 
D_lab =  -100.*u.kilometer 
D_model =  -600.*u.kilometer 

d_moho0 = GEO.nd(D_moho)-dy_nd/4
d_moho1 = GEO.nd(D_moho)+dy_nd/4

from underworld.mesh import FeMesh_IndexSet

condition = np.where((Model.mesh.data[:,1]>= d_moho0)&(Model.mesh.data[:,1]<= d_moho1))
IndexSet = FeMesh_IndexSet(Model.mesh, topologicalIndex=0, size=Model.mesh.nodesGlobal, fromObject=condition[0])

minCoord = tuple([GEO.nd(val) for val in Model.minCoord])
maxCoord = tuple([GEO.nd(val) for val in Model.maxCoord])

init_mesh = uw.mesh.FeMesh_Cartesian(elementType=Model.elementType,
                                    elementRes=Model.elementRes,
                                    minCoord=minCoord,
                                    maxCoord=maxCoord,
                                    periodic=Model.periodic)

TField = init_mesh.add_variable(nodeDofCount=1)
TField.data[:, 0] = init_mesh.data[:, 1].copy()

top = Model.top_wall
bottom = Model.bottom_wall
interface = IndexSet

conditions = uw.conditions.DirichletCondition(variable=TField,indexSetsPerDof=(top + bottom,))
system = uw.systems.SteadyStateHeat(
    temperatureField=TField,
    fn_diffusivity=1.0,
    conditions=conditions)
solver = uw.systems.Solver(system)

if top: 
    kkt = 250 
    nox = len(top) 
    topo_test = np.zeros(nox)
    v_or_t,all0,ramp0,boxl = 5., 40, 20., 600. 
    pi = np.pi
    d = KL.m
    xcoordi =  init_mesh.data[top][:,0]
    topo_s0 = np.zeros(nox)
    acosine = np.zeros((nox, kkt))
    ff = np.zeros(kkt)
    for k in range(1, kkt+1):
        wavelength1 = 2 * boxl / k/d
        aki = np.float128(2 * pi / wavelength1)

        for ij in range(nox):
            acosine[ij, k-1] = np.cos(xcoordi[ij] * aki)

        ak = boxl / (pi * k)
        ff[k-1] = 2 * v_or_t * ak * ak / (ramp0 * boxl) * (np.cos(all0/ak) - np.cos((ramp0 + all0)/ak))

        for ij in range(nox):
            topo_s0[ij] += ff[k-1] * acosine[ij, k-1]/d
    TField.data[top, 0] = topo_s0

    
comm.Barrier()
TField.syncronise()

#TField.data[top, 0] = topo_s0
# TField.data[axis_lc, 0] = y_lc
# TField.data[axis_ml, 0] = y_ml 

solver.solve()
with Model.mesh.deform_mesh():
     Model.mesh.data[:, -1] = TField.data[:, 0].copy()


# In[12]:


# from underworld import visualisation as vis
# Fig = vis.Figure(resolution=(1200,600),rulers=True,margin = 60)
# Fig.Mesh(Model.mesh)
# Fig.show()
# Fig.save('mesh0.png') 


# In[13]:


from underworld.swarm import Swarm
from collections import OrderedDict
Model.swarm_variables = OrderedDict()

Model.swarm = Swarm(mesh=Model.mesh, particleEscape=True)
Model.swarm.allow_parallel_nn = True
if Model.mesh.dim == 2:
    particlesPerCell = GEO.rcParams["swarm.particles.per.cell.2D"]
else:
    particlesPerCell = GEO.rcParams["swarm.particles.per.cell.3D"]

Model._swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(
    swarm=Model.swarm,
    particlesPerCell=particlesPerCell)

Model.swarm.populate_using_layout(layout=Model._swarmLayout)

Model._initialize()


# In[14]:


lc_Shape = GEO.shapes.Layer(top=5*u.kilometer, bottom=D_moho)
ml_Shape = GEO.shapes.Layer(top=D_moho, bottom=D_lab)
ma_Shape = GEO.shapes.Layer(top=D_lab, bottom=D_model)

#air = Model.add_material(name="Air", shape=air_Shape)
lc = Model.add_material(name="Crust", shape=lc_Shape)
ml = Model.add_material(name="Mantle Lithosphere", shape=ml_Shape)
ma = Model.add_material(name="Mantle Asthenosphere", shape=ma_Shape)

#Sediment = Model.add_material(name="Sediment")


# In[15]:


#npoints = nox*2-1
# coords = np.ndarray((npoints, 2))
# coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
# coords[:, 1] = H0*np.exp(-np.power(coords[:, 0] - mu, 2.) / gd)
# Model.add_passive_tracers(name="Surface", vertices=coords)

coords = np.ndarray((npoints, 2))
coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
coords[:, 1] = GEO.nd(D_moho)
Model.add_passive_tracers(name="Moho", vertices=coords)

coords = np.ndarray((npoints, 2))
coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
coords[:, 1] = GEO.nd(D_lab)
Model.add_passive_tracers(name="LAB", vertices=coords)


# In[16]:


# #if uw.mpi.rank == 0:
# from underworld import visualisation as vis
# fig_res = (1200,600)

# Fig = vis.Figure(resolution=fig_res,rulers=True,margin = 80,rulerticks=7,quality=3,clipmap=False)
# #Fig.Points(Model.Surface_tracers, pointSize=4.0)
# Fig.Points(Model.Moho_tracers, pointSize=4.0)
# Fig.Points(Model.LAB_tracers, pointSize=4.0)
# Fig.Points(Model.swarm, Model.materialField,fn_size=2.0,discrete=True,colourBar=False)
# Fig.show()
# Fig.save("Modelsetup.png")


# In[17]:


lc.density = 2800. * u.kilogram / u.metre**3 
ml.density = 3300. * u.kilogram / u.metre**3 
ma.density = 3300. * u.kilogram / u.metre**3 

Model.minViscosity = 1e18 * u.pascal * u.second
Model.maxViscosity = 1e26 * u.pascal * u.second

lc.viscosity = 1e24 * u.pascal * u.second
ml.viscosity = 1e21 * u.pascal * u.second
ma.viscosity = 1e21 * u.pascal * u.second

dt_e = 3000 * u.year   # observation time


# In[18]:


mu      = 1.5e11 * u.pascal # Shear Modulus
#eta     = 1e23 * u.pascal * u.second 
#alpha   = eta / mu       # Relaxation time
#print('Relaxation time = ', alpha.to(u.years))

lc.elasticity = GEO.Elasticity(shear_modulus= mu,observation_time= dt_e)    
ml.elasticity = GEO.Elasticity(shear_modulus= mu,observation_time= dt_e)
ma.elasticity = GEO.Elasticity(shear_modulus= mu,observation_time= dt_e)         


# In[19]:


if type_bcs == "_FreeSlipBot":
    Model.set_velocityBCs(left = [0.,None],
                      right=[0., None],
                      top = [None, None],
                        bottom = [None,0.],order_wall_conditions=["left","right","top","bottom"])

Model.init_model(pressure="lithostatic")
#Model.freeSurface = True

#from _freesurface_moho import FreeSurfaceProcessor
#Model.inter_wall = IndexSet
Model.freeSurface = True
#Model._freeSurface = FreeSurfaceProcessor(Model)


# In[20]:


if type_bcs == "_LithoBot":
    Model.init_model(pressure="lithostatic")
    Model.freeSurface = True

    Model.set_velocityBCs(left = [0.,0.],
                      right=[0., 0.],
                      top = [None, None],
                      bottom = [None,None],order_wall_conditions=["left","right","top","bottom"])

    def update_traction():
        tmp1 = uw.utils.Integral(fn=Model._densityFn*GEO.nd(gravity),mesh=Model.mesh, integrationType='volume')
        tmp2 = uw.utils.Integral(fn=1.,mesh=Model.mesh, integrationType='Surface',surfaceIndexSet=Model.bottom_wall)
        pbot = tmp1.evaluate()[0]/(tmp2.evaluate()[0])
        #GEO.dimensionalise(pbot,u.megapascal)
        #lithopT = Model.tractionField.evaluate(Model.bottom_wall)[:,1]
        #GEO.dimensionalise(lithopT.mean(),u.megapascal)

        Model.tractionField.data[:,1] = pbot
        Model.set_stressBCs(bottom=[None,Model.tractionField])

    update_traction()

#Model.pre_solve_functions["B-traction"] = update_traction


# In[21]:


Model.temperatureDot = Model._temperatureDot


# In[ ]:


Total_Time = 1010*u.kiloyear
checkpoint_interval = 10*u.kiloyear
dt = 1*u.kiloyear
Model.run_for(Total_Time, checkpoint_interval=checkpoint_interval,dt=dt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





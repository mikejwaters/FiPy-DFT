



lz = 3.0 # Ang
lx = 4.0
ly = 3.0

dx = 0.03
dy = 0.03
dz = 0.03 # Ang

nz = int(lz/dz)
ny = int(ly/dy)
nx = int(lx/dx)

lz = nz*dz
ly = ny*dy
lx = nx*dx
	
from mpi4py import MPI
m4comm = MPI.COMM_WORLD
if m4comm.Get_rank() == 0:
	print 'nz', nz ,'dz', dz, 'lz', lz 
	print 'ny', ny ,'dy', dy, 'ly', ly 
	print 'nx', nx ,'dx', dx, 'lx', lx 






number_of_electrons = 4


steps = 100


accuracy = 10.0**-5
phi_solver_iterations_per_step = 1000
save_infrequency = 5
initial_solver_iterations_per_step = 7
###
read_restart = True
GC = True # does garbage collection at each self-consistent iteration
use_trilinos = True
######################## Constants!


qe = -1.0 # electron charge
a0 = 0.52917721067 # codata 2014 in Ang

hbar = 1.054571800 * 10**-34 #(SI)
me0  = 9.10938356 * 10**-31 #(SI)
eV = 1.6021766208 * 10**-19 #(SI)

tcoeff = (hbar**2)/(2.0*me0)*(1.0/eV)*(10.0**10)**2 # = hbar^2/(2*Me) in units of [eV*Ang^2]

from numpy import pi


eu = 1.6021766208 * 10**-19 #electron charge (SI)
epsilon0 = (8.854187817 * 10**-12) # in SI
epsilon0 = epsilon0 * (1.0/eu)*(1.0/10.0**10) # conversion
cc0 = 1.0/(4.0*pi*epsilon0) # about 14.4 eV*Ang usually used in atomistics

###########3


from fipy import __version__ as FiPy_version
if m4comm.Get_rank() == 0:
	print 'FiPy Version:', FiPy_version
################
from fipy import *
from numpy.random import rand
from numpy import sqrt, savetxt, array, exp, arange, zeros, loadtxt, sign
import os.path
if GC: import gc

if m4comm.Get_rank() == 0: print 'Creating mesh and cell variables'

mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz,)# origin = (lx/2.0, ly/2.0, lz/2.0))
Vcores = CellVariable(name = 'Core Potential for Electrons', mesh=mesh, value = 0.0)
rho_e = CellVariable(name = 'Electron Density', mesh=mesh, value = 0.0)
phi_e = CellVariable(name = 'Hartree Potential', mesh=mesh, value = 0.0)



### choosing a solver
if use_trilinos:
	from fipy.solvers.trilinos import LinearGMRESSolver 				as phi_method
	from fipy.solvers.trilinos.preconditioners import MultilevelSGSPreconditioner 	as phi_precon

	#from fipy.solvers import trilinos as trisol
	#print dir(trisol)
	#from fipy.solvers.trilinos import DefaultAsymmetricSolver 	as psi_method # this is just GMRES...
	#from fipy.solvers.trilinos import DefaultSolver 		as psi_method
	#from fipy.solvers.trilinos import DummySolver 			as psi_method
	#from fipy.solvers.trilinos import GeneralSolver 		as psi_method
	from fipy.solvers.trilinos import LinearBicgstabSolver		as psi_method # works well with Jacobi, 20 iterations
	#from fipy.solvers.trilinos import LinearCGSSolver 		as psi_method # DNF with Jacobi, might need reduced minimum mixing paramter
	#from fipy.solvers.trilinos import LinearGMRESSolver 		as psi_method # works well with Jacobi, 32 iterations, optimal mixer might need to change iterations faster for it to keep up with the needs of the GMRES solver, looks like it could have done it in 21
	#from fipy.solvers.trilinos import LinearLUSolver 		as psi_method # DNF with Jacobi
	#from fipy.solvers.trilinos import LinearPCGSolver 		as psi_method # standard, 23 iterations
	#from fipy.solvers.trilinos import TrilinosMLTest		as psi_method # doesn't seem to be anything

	##### choosing a preconditioner
	#from fipy.solvers.trilinos import preconditioners as precons
	#print dir(precons)
	#from fipy.solvers.trilinos.preconditioners import DomDecompPreconditioner 		as psi_precon # NotImplementedError
	#from fipy.solvers.trilinos.preconditioners import ICPreconditioner 			as psi_precon # segfaults... with permisions errors
	from fipy.solvers.trilinos.preconditioners import JacobiPreconditioner 		as psi_precon # standard.
	#from fipy.solvers.trilinos.preconditioners import MultilevelDDMLPreconditioner		as psi_precon # bizzare warning about compilation..., DNF with LinearBicgstab
	#from fipy.solvers.trilinos.preconditioners import MultilevelDDPreconditioner 		as psi_precon # unstable
	#from fipy.solvers.trilinos.preconditioners import MultilevelNSSAPreconditioner 	as psi_precon # unstable
	#from fipy.solvers.trilinos.preconditioners import MultilevelSAPreconditioner 		as psi_precon # unstable
	#from fipy.solvers.trilinos.preconditioners import MultilevelSGSPreconditioner 		as psi_precon # unstable
	#from fipy.solvers.trilinos.preconditioners import MultilevelSolverSmootherPreconditioner as psi_precon # Segfaults...

else:
	from fipy.solvers.pysparse import LinearPCGSolver as psi_method
	from fipy.solvers.pysparse import LinearPCGSolver as phi_method
	from fipy.solvers.pysparse import SsorPreconditioner as psi_precon
	from fipy.solvers.pysparse import SsorPreconditioner as phi_precon

psi_solver = psi_method(iterations = initial_solver_iterations_per_step, precon = psi_precon() )
phi_solver = phi_method(iterations = phi_solver_iterations_per_step, precon = phi_precon() )


xc, yc, zc = mesh.cellCenters
dv = dx*dy*dz 
simulation_volume = lx*ly*lz

if m4comm.Get_rank() == 0:
	print 'simulation volume', simulation_volume

############## set initial applied potential
if m4comm.Get_rank() == 0: print "Setting Core Potentials"
atomic_data=loadtxt('input.xyz') 
###this file has  the format
# atomic_number x y z
# e.g. carbon at some coordinates:
# 6 1.0 2.1 1.1
for i in range(atomic_data.shape[0]):

	Z = atomic_data[i][0]
	x = lx/2.0 + atomic_data[i][1]
	y = ly/2.0 + atomic_data[i][2]
	z = lz/2.0 + atomic_data[i][3]
	#print x
	coeff = -Z/(4.0*pi*epsilon0)
	Vcores.setValue(coeff/sqrt( (xc-x)**2 + (yc-y)**2 + (zc-z)**2) + Vcores)

if True: 
	if m4comm.Get_rank() == 0: print 'saving core potentials...'
	savetxt('Vcores.txt', Vcores.globalValue)

########### write cell data

fid = open('cell_data.txt','w')
fid.write('numcells and dx in x direction\n')
fid.write('%i\t%f\n'%(nx,dx))
fid.write('numcells and dy in y direction\n')
fid.write('%i\t%f\n'%(ny,dy))
fid.write('numcells and dz in z direction\n')
fid.write('%i\t%f\n'%(nz,dz))
fid.close()


################## Initialize all psi's
psi_e_list = []
for n in range(number_of_electrons):
	psi_e_list.append(  CellVariable(name = '$\psi_{e%i}$' %n , mesh=mesh, value = 1.0, hasOld = True))

##### setting up charge density
for n in range( number_of_electrons):
	rho_e = rho_e + qe*psi_e_list[n]*psi_e_list[n] 

############## this sets up the electrostatic potential	
phi_e.equation = (0.0 == DiffusionTerm(coeff = epsilon0) + rho_e)



##################### some function definitions

def inner_product(psi_a,psi_b):
	ans = ((psi_a * psi_b).cellVolumeAverage * simulation_volume).value
	
	return ans

def normalize(psi):	
	a = 1.0/sqrt( inner_product(psi,psi) ) # normalizing factor#
	#print 'normalizing factor',a 
	psi.setValue( psi*a)

def num_electrons(psi):
	return inner_product(psi,psi)

def sandwich_H(psi_a, psi_b):  # the hamiltonian is entered twice in this simulation, once as the equation to be solved and here as a function to be evaluated
	Hpsi = -tcoeff * psi_b.faceGrad.divergence + (Vcores + qe*phi_e)*psi_b
	return ((psi_a*Hpsi).cellVolumeAverage * simulation_volume).value

def H_expect(psi):
	return sandwich_H(psi, psi)

def subtract_projected_on_normed(psi_b,psi_a):
	coef=inner_product(psi_a,psi_b)
	psi_b.setValue(psi_b - psi_a*coef)


def kinetic_energy_e(psi):
	Tpsi = -tcoeff* psi.faceGrad.divergence
	return ((psi*Tpsi).cellVolumeAverage * simulation_volume)
	
##### initialization of wave functions ##############

if os.path.isfile('phi_e.txt') == False:
	read_restart = False


if read_restart:
	if m4comm.Get_rank() == 0: print 'Reading restart'
	## electrons
	for n in range(number_of_electrons):
		if os.path.isfile('psi_e_%i.txt'%n):
			if m4comm.Get_rank() == 0: print 'Reading electron wave function %i'%n
			input_data = loadtxt('psi_e_%i.txt'%n)
			psi_e_list[n].setValue(input_data)
			psi_e_list[n].updateOld()
		else: # does random initialization
			if m4comm.Get_rank() == 0: print 'Random initialization for missing electron wave functions %i'%n
			psi_e_list[n].setValue(rand(nx*ny*nz)-0.5)
			

		
	## phi_e
	if m4comm.Get_rank() == 0: print 'Reading electron potential...'
	input_data = loadtxt('phi_e.txt')
	phi_e.setValue(input_data)

	
else:
	if m4comm.Get_rank() == 0: print 'Guess initialization'
	# for all psi's
	for n in range(number_of_electrons):
		psi_e_list[n].setValue(rand(nx*ny*nz)-0.5)

	phi_e.setValue(rand(nx*ny*nz))



############## orthonormalize functions then normalize
if m4comm.Get_rank() == 0: print 'Orthonormalizing...'
if number_of_electrons > 0:
	normalize(psi_e_list[0]) #normalize the lowest energy electron
	if number_of_electrons >1 :normalize(psi_e_list[1]) # only if there is second electron
	
	for n in range(2, number_of_electrons):
		for lower_wave in range(n%2,n,2): #since range goes from 0 to n-1, skipping opposite-spin electrons
			subtract_projected_on_normed(psi_e_list[n],psi_e_list[lower_wave])
			normalize(psi_e_list[n])

### checking normalization
if False:
	for n in range(number_of_electrons):
		num = num_electrons(psi_e_list[n])
		if m4comm.Get_rank() == 0: print n, num








####### initialize boundary conditions
from fipy.boundaryConditions.constraint import Constraint

if m4comm.Get_rank() == 0: print "applying BC's"
constraint_list_e = []
for n in range(number_of_electrons):
	constraint_list_e.append(  Constraint(0, mesh.exteriorFaces))
	psi_e_list[n].constrain(constraint_list_e[n])


phi_e.constrain(Constraint(0.0, mesh.facesBack))
phi_e.constrain(Constraint(0.0, mesh.facesFront))

phi_e.faceGrad.constrain( 0.0, mesh.facesRight)
phi_e.faceGrad.constrain( 0.0, mesh.facesLeft)
phi_e.faceGrad.constrain( 0.0, mesh.facesTop)
phi_e.faceGrad.constrain( 0.0, mesh.facesBottom)


############ initialize energies ... <psi_n|H|psi_n> = E_n
energy_list_e = []
solver_iterations_per_step_e = []
rel_error_e = []
for n in range(number_of_electrons):
	e = H_expect(psi_e_list[n])
	energy_list_e.append(Variable(value=e) )
	solver_iterations_per_step_e.append(initial_solver_iterations_per_step)
	rel_error_e.append(1.0)


Te = 0.0 # total kinetic energy
for n in range(number_of_electrons):
	Te += kinetic_energy_e(psi_e_list[n])
toten_old = Te - inner_product(rho_e,Vcores) # total energy, rho is charge density and is always negative because electrons


if m4comm.Get_rank() == 0: print '############### Starting Self-Consistent Iteration ###########'
if m4comm.Get_rank() == 0: print 'Initial Total Electron Energy', toten_old 


if read_restart:
	mode = 'a'
else: 
	mode = 'w'
fid = open('iteration_energies.log', mode)
fid_rel_error = open('relative_errors.log', mode)



#### HAMILTONIAN and Schroedinger equation!
for n in range(0, number_of_electrons):
	psi_e_list[n].equation = (0 == DiffusionTerm(-tcoeff) + (Vcores + qe*phi_e - energy_list_e[n]) * psi_e_list[n] )

max_rel_error = 1.0
step = 1
while step<=steps and max_rel_error > accuracy:


	
	if m4comm.Get_rank() == 0: print 'Solving Phi e'
		
	phi_e_res = phi_e.equation.sweep(var = phi_e, solver = phi_solver )
	
	if m4comm.Get_rank() == 0: print 'phi_e_res' , phi_e_res
	
	
	##############Solving the ELECTRON wavefunctions with orthonormalization after each solve 
	for n in range(0, number_of_electrons):
		if rel_error_e[n] < accuracy:
			if m4comm.Get_rank() == 0: print  '--- Skipping electron %i because it is converged' %n
		else:
			
			psi_e_list[n].updateOld() ## update before solving for the next wavefunctions
		
			if m4comm.Get_rank() == 0: print '--- Working on electron %i' %n
			if m4comm.Get_rank() == 0: print 'Solving...'
			
			#### Solving it
			psi_solver.iterations = solver_iterations_per_step_e[n]
			psi_e_res = psi_e_list[n].equation.sweep(var = psi_e_list[n], solver = psi_solver )
			if m4comm.Get_rank() == 0: print 'psi_e_res', psi_e_res
			
			
			##############
			if m4comm.Get_rank() == 0: print 'Orthonormalizing...'
			normalize(psi_e_list[n])
			for lower_wave in range(n%2,n,2): #only normalizes against the spin up or spin down electrons
				subtract_projected_on_normed(psi_e_list[n],psi_e_list[lower_wave])
				normalize(psi_e_list[n])
			
			
			Eold = float(energy_list_e[n].value) # evaluates the Variable
			Enew = H_expect(psi_e_list[n])
			energy_list_e[n].setValue( Enew )
				
			rel_error_e[n] = abs((Enew-Eold)/Enew)
		


	max_rel_error = rel_error_e[0] # finds the largest relative error
	for n in range(number_of_electrons):
		if rel_error_e[n] > max_rel_error:
			max_rel_error = rel_error_e[n]
			
	

	
	########## Screen and log output, everything below here is not very iteresting.
	if m4comm.Get_rank() == 0: 
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print '\nStep', step, 

	Te = 0.0
	for n in range(number_of_electrons):
		Te += kinetic_energy_e(psi_e_list[n])
	Ecores = -inner_product(rho_e,Vcores)
	toten = Te + Ecores	
	if m4comm.Get_rank() == 0: 
		print 'Total Energy' , toten,

	de = toten - toten_old
	
	if m4comm.Get_rank() == 0: 
		print 'dE', de,
		print 'dE/|toten|' , de/abs(toten),
		print 'max_rel_error', max_rel_error
	
	toten_old = toten
	if m4comm.Get_rank() == 0: 
		print '\nElectron Energies', energy_list_e
		fid.write('\nStep ' + str(step)+ ' Total Energy ' + str(toten) +' dE '+ str(de) + ' dE/|toten| ' + str(de/abs(toten) ) + ' Max Rel Error ' +str(max_rel_error) )#+ '\nHole Energies '+ str(energy_list_h[0]))
	
	
	
	
	if m4comm.Get_rank() == 0: # this section saves file output
		def str_list(a):
			output = ''
			for thing in a:
				output = output + str(thing) + ' '
			return output
	
		fid_rel_error.write(str_list(rel_error_e) + '\n')

		fid_rel_error.flush()

	
	
	############# This output the energy eigenvalues
	if m4comm.Get_rank() == 0:
		e_energy_string  = ''
		for n in range(number_of_electrons):
			e_energy_string = e_energy_string+' '+str(float(energy_list_e[n].value))
		fid.write('\nElectron Energies ' + e_energy_string)
	
	
	#############
	
	hartree_energy = 0.5*inner_product(rho_e,phi_e)

	if m4comm.Get_rank() == 0: 
	
		fid.write('\nhartree_energy ' + str(hartree_energy) )
		print 'hartree_energy ' + str(hartree_energy)
		
		fid.write('\nTe ' + str(Te) )
		print 'Te ' + str(Te) 
	
		fid.write('\nEcores ' + str(Ecores) )
		print 'Ecores ' + str(Ecores)
	
	fid.flush()
	
	#### saving output### 
	savetxt('Energy_eigenvalues_e.txt',array(energy_list_e))
	if step%save_infrequency == 0:
			if m4comm.Get_rank() == 0: print 'Saving Cell data...'
			for n in range( number_of_electrons):
				savetxt('psi_e_%i.txt'%n,psi_e_list[n].globalValue)
			savetxt('phi_e.txt',phi_e.globalValue)
			savetxt('rho_e.txt',rho_e.globalValue)
	
	if GC:
		if m4comm.Get_rank() == 0: print 'Collecting Garbage...'
		gc.collect()
		
	step +=1 ## loop ends here!
	
	

#### saving output### 
 
savetxt('Energy_eigenvalues_e.txt',array(energy_list_e))


if m4comm.Get_rank() == 0: print 'Finished!, Saving Cell data...'
for n in range( number_of_electrons):
	savetxt('psi_e_%i.txt'%n,psi_e_list[n].globalValue)

savetxt('phi_e.txt',phi_e.globalValue)
savetxt('rho_e.txt',rho_e.globalValue)


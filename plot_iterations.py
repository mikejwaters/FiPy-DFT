from pylab import *


e = []
max_rel_error = []
alpha = []
solver_iterations_per_step = []

fid = open('iteration_energies.log','r')

for line in fid:
	if 'Total Energy' in line:
		sline = line.split()
		e.append(float(sline[4]))		
		max_rel_error.append(float(sline[-1]))	

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, figsize = (8,8))
		
ax1.plot(e, label = 'toten new')

ax1.legend()
ax1.grid(True)

############
data = loadtxt('relative_errors.log').T

if len(data.shape) > 1:
	for i in range(data.shape[0]):
		ax2.semilogy(data[i])
else:
	ax2.semilogy(data)
ax2.grid(True)

###########

###########
tight_layout()

show()

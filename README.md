# FiPy-DFT 

A simple and easy to modify Density Functional Theory (DFT) code built on top of the finite volume package [FiPy](http://www.ctcms.nist.gov/fipy/).

### An Overview ###

This is not a fast DFT code nor is it particularly accurate. What is it? It is a very easy to modify code. Of the common theoretical features in electronic structure code, basically none are implemented. This is deliberate. With nothing implemented, there less structural barriers to implementing something new. With that in mind, it is easier to describe what is implemented rather than what is not.

* The effective single particle Hamiltonian contains only the Coulomb attraction to nuclear cores and repulsion from the total electron density.
* All electrons are treated explicitly. Spin up and down electrons are simply orthonormalized separately but otherwise treated equally. 
* Wave functions and energy eigenvalues are calculated using self-consistent iteration with mixing between old and new wave function solutions for improved  numerical stability. 
* The code does not use basis sets in the traditional sense but instead uses a tessellated finite volume mesh as a 'basis'. Currently, it's just a regular 3D grid. This removes a theoretical complexity from the code. For all systems, regardless of Hamiltonian, a mesh only needs to be fine enough for the desired energy accuracy. 
* Previous results are saved, and attempts are made to restart previous calculations.
* Can be run in parallel using MPI with the PyTrilinos backend for FiPy.

### Running ###

To run, simply execute the python script:

    $ python fipy-dft.py
   
To run in parallel on 4 cores:
  
    $ mpirun -np 4 python fipy-dft.py

To run in parallel on all cores:

    $ mpirun -np $(nproc) python fipy-dft.py

### Issues ###

* Running in parallel results in memory leaks for some people. Output data often. A thorough investigation is required. 
* Because electron-electron interactions are calculated using the Poisson equation, the boundary conditions must apply at least one fixed value (Dirichlet) boundary condition on the electrostatic potential from the total electron charge density. Currently, the best solution is to give extra in room the *z* direction till energies reasonably converge. For non-periodic calculations, the best solution may be to use a method like [this](http://cseweb.ucsd.edu/groups/hpcl/scg/papers/2005/hpsec05-scalable-poisson.pdf). For periodic calculations, which physically require zero net charge, it may be worth adding the nuclear charges to the mesh and then offloading the electrostatics problem to an FFT based solver. 

### History ###

This code was created as an offshoot of a code for calculating states with potential barriers and applied fields. This is why the current boundary conditions are so unusual. When the paper is published, I’ll try to link to it here so that the motivation is clear.

import numpy as np 
import entities               # module contains mathematical objects such as functions, processes and model.  
import finmath_calculations   # module contains matrix constructions, operators and other numerical calculations. 

# SIMPLE EULER ICREMENTS. ##############################################################################################

def dNu(dt, prev_t, prev_Nu, Nu_SDE, dB):
    '''
    1-d function.
    
    PARAMS:
    dt      -- size of the time step.
    prev_t  -- time point at which we freeze the coefficient.
    prev_Nu -- Nu_t
    Nu-SDE  -- the class of the volatility process that we are approximating. 
    dB      -- the corresponding Brownian increment. 
    
    RETURNS:
    The approximated value of volatility increment on time: [t, t + dt]. 
    '''
    return Nu_SDE.drift_coef.value(prev_t, prev_Nu) * dt + Nu_SDE.diffusion_coef.value(prev_t, prev_Nu) * dB
    

def drho(dt, prev_Nu_vector, prev_rho, rho_SDE, dWhat):
    '''
    Logic is the same.
    '''
    return rho_SDE.drift_coef.value(prev_Nu_vector, prev_rho) * dt + rho_SDE.diffusion_coef.value(prev_rho) * dWhat
    

def dS(dt, prev_t, prev_S, prev_Nu, S_SDE, dW):
    '''
    Logic remains the same. 
    '''
    return S_SDE.drift_coef.value(prev_t, prev_S) * dt
    + S_SDE.diffusion_coef.value(prev_t, prev_S, prev_Nu) * dW

########################################################################################################################


# SIMPLE SCHEME ########################################################################################################

def SimpleEulerMemoryInitialization(Memory, S0, Nu0, rho0):
    Memory.T[0] = np.concatenate((S0, Nu0, np.array([rho0])), axis=0)
   

def SimpleEulerSimulation(model, dt, n, S0, Nu0, rho0):
    '''
    PARAMS:
    model -- StochasticCorrModel that contains all processes and the correlation matrix. 
    dt    -- time step size
    n     -- number  of iterations. ndt = T if we are simulating on the interval [0, T]. 
    '''
    # Array that contains all trajectories. Initialization. 
    Memory = np.zeros((2*model.N + 1) * (n + 1)).reshape(2*model.N + 1, n + 1) 
    SimpleEulerMemoryInitialization(Memory, S0, Nu0, rho0)
    
    for iteration in range(n):
        Gaussian_increments = CalculateDiffusionIncrements(model.GeneralCorrMatrix(Memory[-1][iteration]),
                                                           dt, model.N) 
        
        for i in range(model.N):        # Generate volatilities.
            Memory[i][iteration + 1] = Memory[i][iteration] + dS(dt, iteration*dt, Memory[i][iteration], 
                                                              Memory[model.N + i][iteration], 
                                                              model.particles[i], Gaussian_increments[i])
            
        for i in range(N, model.N * 2): # Generate particles. 
            Memory[i][iteration + 1] = Memory[i][iteration] + dNu(dt, iteration*dt, Memory[i][iteration], 
                                                                  model.vols[i], Gaussian_increments[i])
            
        Memory[-1][iteration + 1] = Memory[-1][iteration] + drho(dt, 
                                                                 Memory[model.N:model.N*2].T[iteration],
                                                                 Memory[-1][iteration],
                                                                 model.corr_process, 
                                                                 Gaussian_increments[-1])  
        
    return Memory

########################################################################################################################
import numpy as np 
import entities



# MATRIX OPERATIONS ####################################################################################################

def ConstructParticleCorrelationMatrix(rho, N):
    '''
    CAN BE OPTIMIZED!!!
    
    
    PARAMS:
    rho -- correlation process value at time t. 
    N   -- number of particles in our model.
    
    RETURNS:
    The simple particle-particle correlation matrix with correlation value rho. 
    Is further used in constructing the general correlation matrix. 
    '''
    rho_vector = rho * np.ones(N).reshape(N, 1)
    ones_vector = np.ones(N).reshape(N, 1)
    
    return (rho_vector @ ones_vector.T) - np.diag(np.ones(N) * rho) + np.diag(np.ones(N))
    

def CalculateGeneralCorrMatrix(rho, rho_skew, ParticleVolCorrMatrix, VolCorrMatrix, N):
    '''
    Calculates the correlation matrix of the whole model. This matrix describes all of the 
    crosscorrelations between brownian motions and is further used for the calculation of
    the brownian increments of the model. 
    
    PARAMS:
    rho                   -- correlation process value at time t. 
    rho_skew              -- correlation between any brownian motion with the motion of the correlation process. 
    ParticleVolCorrMatrix -- dW^idB^j/dt are the elems of this one. 
    VolCorrMatrix         -- dB^idB^j are the elems of that one. 
    N                     -- number of particles in the model. 
    
    RETURNS:
    2N+1 x 2N+1 correlation matrix. 
    '''
    
    ParticleCorrMatrix = ConstructParticleCorrelationMatrix(rho, N)
    L1 = np.concatenate((ParticleCorrMatrix, ParticleVolCorrMatrix), axis=1)
    L2 = np.concatenate((ParticleVolCorrMatrix, VolCorrMatrix), axis=1)
    
    MainBlock = np.concatenate((L1, L2), axis=0)
    skew_vector = np.ones(2*N).reshape(2*N, 1) * rho_skew
    
    ExtendedMainBlock = np.concatenate((MainBlock, skew_vector), axis=1)
    extended_skew_vector = np.concatenate((skew_vector, np.ones(1).reshape(1, 1)), axis=0)
    
    return np.concatenate((ExtendedMainBlock, extended_skew_vector.T), axis=0)


def CalculateDiffusionIncrements(CorrMatrix, dt, N):
    '''
    Given the sqare root of the correlation matrix and the size of time step, 
    calculate the values of dW_t or dB_t for all diffusion brownian motions in our model. 
    
    PARAMS:
    
    CorrMatrix: value of stochastic correlation matrix observation at time t.
    Size of the matrix is 2N+1 x 2N + 1. 
    
    dt: number, the size of time step of the simulation. 
    
    N:  the number of particles in our system. 
    '''
    iidStandartNormalVector = np.random.randn(2*N + 1)
    Increment = np.linalg.cholesky(CorrMatrix) @ iidStandartNormalVector * np.sqrt(dt)
    
    return Increment

########################################################################################################################



# ITO OPERATORS ########################################################################################################

def ParticleDriftItoOperator(ParticleProcess, t, S, Nu):
    
    '''
    Function that returns values of two operators that are used in Ito-Doeblin formula. 
    The value is calculated in the point (t, S, Nu). 
    
    So, particle drift mu(t, S) can be approximated as:
    dmu(t, D) = DriftValue(t, S, Nu)*dt + DiffusionValue*dWt,
    where dt, dWt are taken from dSt. 
    
    PARAMS: 
    ParticleProcess: class of an SDE that describes our particle. 
    t: time moment
    S: particle value at time t.
    Nu: volatility value at time t. 
    
    RETURNS: 
    Two numbers, the values of the corresponding operators. 
    '''  
    DriftValue = ParticleProcess.drift_coef.dt(t, S) + 
                 ParticleProcess.drift_coef.value(t, S) * ParticleProcess.drift_coef.dx(t, S) +
                 0.5 * (ParticleProcess.diffusion_coef(t, S, Nu) ** 2) * ParticleProcess.drift_coef.dxdx(t,S)
            
    DiffusionValue = ParticleProcess.diffusion_coef(t, S, Nu) * ParticleProcess.drift_coef.dx(t, S)
    
    return DriftValue, DiffusionValue


def ParticleDiffusionItoOperator(ParticleProcess, VolatilityProcess, Corr, t, S, Nu):
    '''
    PARAMS:
    ParticleProcess: an SDE class that describes our asset price. 
    VolatilityProcess: an SDE class that describes the volatility of the asset price. 
    t: current time moment
    Corr: correlation between dWt and dBt: dWtdBt = Corr * dt at time point t. 
    S: asset price at time t
    Nu: volatility value at time t. 
    
    RETURNS:
    DriftValue -- the value of the drift operator at time t. 
    DiffusionSigmaValue, DiffusionDvalue -- diffusion operators values at time t. 
    '''
    DriftValue = ParticleProcess.diffusion_coef.dt(t, S, Nu) + 
                 ParticleProcess.drift_coef.value(t, S) * ParticleProcess.diffusion_coef.dx(t, S, Nu) + 
                 VolatilityProcess.drift_coef.value(t, Nu) * ParticleProcess.diffusion_coef.dy(t, S, Nu) +
                 0.5 * (ParticleProcess.diffusion_coef.value(t, S, Nu)**2) 
                 * ParticleProcess.diffusion_coef.dxdx(t, S, Nu) + 
                0.5 * (VolatilityProcess.diffusion_coef.value(t, Nu)**2) 
                     * ParticleProcess.diffusion_coef.dydy(t, S, Nu) + 
                Corr * ParticleProcess.diffusion_coef.value(t, S, Nu) * 
                VolatilityProcess.diffusion_coef.value(t, Nu) * ParticleProcess.diffusion_coef.dxdy(t, S, Nu)
    
    DiffusionSigmaValue = ParticleProcess.diffusion_coef.value(t, S, Nu) 
                        * ParticleProcess.diffusion_coef.dx(t, S, Nu) 
        
    DiffusionDValue     = VolatilityProcess.diffusion_coef.value(t, Nu)  
                        * ParticleProcess.diffusion_coef.dy(t, S, Nu)
    
    return DriftValue, DiffusionSigmaValue, DiffusionDValue
    

def VolatilityDriftItoOperator(VolatilityProcess, t, Nu):
    '''
    Here, logic remains the same with the previous operators. This one
    has a less complicated structure.  
    '''
    DriftValue = VolatilityProcess.drift_coef.dt(t, Nu) +  
                 VolatilityProcess.drift_coef.value(t, Nu) * VolatilityProcess.drift_coef.dx(t, Nu)  +
                 0.5 * (VolatilityProcess.diffusion_coef.value(t, Nu) ** 2) 
                 * VolatilityProcess.drift_coef.dxdx(t, Nu)
    
    DiffusionValue = VolatilityProcess.diffusion_coef.value(t, Nu) * VolatilityProcess.drift_coef.dx(t, Nu)
    
    return DriftValue, DiffusionValue
    
    
def VolatilityDiffusionItoOperator(VolatilityProcess, t, Nu):
    '''
    Here, logic remains the same with the previous operators. This one
    has a less complicated structure.  
    '''
    DriftValue = VolatilityProcess.diffusion_coef.dt(t, Nu) +  
                 VolatilityProcess.drift_coef.value(t, Nu) * VolatilityProcess.diffusion_coef.dx(t, Nu)  +
                 0.5 * (VolatilityProcess.diffusion_coef.value(t, Nu) ** 2) 
                 * VolatilityProcess.diffusion_coef.dxdx(t, Nu)
    
    DiffusionValue = VolatilityProcess.diffusion_coef.value(t, Nu) * VolatilityProcess.diffusion_coef.dx(t, Nu)
    
    return DriftValue, DiffusionValue
                
def CorrelationDriftItoOperator(CorrelationProcess, VolatilityVectorProcess, t, NuVector, rho, N):
    '''
    '''
    DriftValue = (- CorrelationProcess.alpha) * CorrelationProcess.drift_coef.value(NuVector, rho)
    for k in range(N):
        DriftValue += VolatilityVectorProcess[k].drift_coef.value(t, NuVector[k]) *
                      CorrelationProcess.drift_coef.dx(NuVector, rho, k)       
    for k in range(N):
        for l in range(N):
            DriftValue += 0.5 * VolatilityVectorProcess[k].diffusion_coef.value(t, NuVector[k]) 
                       * VolatilityVectorProcess[l].diffusion_coef.value(t, NuVector[l]) 
                       * CorrelationProcess.drift_coef.dxdx(NuVector, rho, k, l)
                    
    DiffusionDVectorValue = []
    for k in range(N):
        DiffusionDVectorValue.append(VolatilityVectorProcess[k].diffusion_coef.value(t, NuVector[k]) 
                                     * CorrelationProcess.drift_coef.dx(NuVector, rho, k))
    
    DiffusionOmegaVectorValue = - CorrelationProcess.diffusion_coef.value(rho)*CorrelationProcess.alpha
    
    return DriftValue, DiffusionDVectorValue, DiffusionOmegaVectorValue
                        
    
def CorrelationDiffusionItoOperator(CorrelationProcess, NuVector, rho):
    '''
    '''    
    DriftValue = CorrelationProcess.drift_coef.value(NuVector, rho) * CorrelationProcess.diffusion_coef.dx(rho) +
                 0.5 * (CorrelationProcess.diffusion_coef.value(rho) ** 2) 
                 * CorrelationProcess.diffusion_coef.dxdx(rho)
            
    DiffusionValue = CorrelationProcess.diffusion_coef.value(rho) * CorrelationProcess.diffusion_coef.dx(rho)
    
    return DriftValue, DiffusionValue

########################################################################################################################



# INTEGRAL APPROXIMATIONS ##############################################################################################
# Maybe this part should be included in approx schemes in second refinement. 

def DoubleIntegral(BrownianIncrements, V, dt, i, j):
    '''
    Calculates the values of double integrals that appear in second-order refinement.
    
    PARAMS:
    BrownianIncrements: vector of all Brownian increments. 
    V: matrix of helping variables. V[i][j] = -V[j][i] for all i < j
       V[i][i] = h for all i. 
       V[i][j] = h with prob 0.5 and -h otherwise for all i < j.
    dt: size of time step
    i, j: number of corresponding Brownian motions with which the integral appeared. 
    '''
    if   (i == 0 and j == 0):
        return 0.5 * dt ** 2
    elif (i == 0):
        0.5 * dt * BrownianIncrements[j]
    elif (j == 0):
        0.5 * dt * BrownianIncrements[i]
    elif (i > 0 and j > 0):
        return 0.5 * (BrownianIncrements[i] * BrownianIncrements[j] - V[i][j])
    else: 
        print("Error in i, j.\n")

########################################################################################################################
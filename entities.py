import numpy as np 
import finmath_calculations

# MATHEMATICAL FUNCTIONS ###############################################################################################

class coef_function1d(object):
    '''
    1-dimensional coeffitient function. It depends on only 1 argument.
    Contains its value, first and second derivatives in point x. 
    '''
    
    def __init__(self, value, FirstDerivative, SecondDerivative):
        self.value = value
        self.dx    = FirstDerivative
        self.dxdx  = SecondDerivative
    
    def ReturnValue(self, x):
        return self.value(x)
    
    def dx(self, x):
        return self.FirstDerivative(x)
    
    def dxdx(self, x):
        return self.SecondDerivative(x)
    
    
class coef_function2d(object):
    '''
    2-dimensional coefficient function. Usually takes
    time-moment and process value as arguments. Both are real numbers.
    '''
    
    def __init__(self, value, Dt, Dx, DxDx):
        self.value = value
        self.Dt    = Dt
        self.Dx    = Dx
        self.DxDx  = Dxdx
    
    def value(self, t, x):
        return value(t, x)

    def dt(self, t, x):
        return self.Dt(t, x)
    
    def dx(self, t, x):
        return self.Dx(t, x)
    
    def dxdx(self, t, x):
        return self.DxDx(t, x)
    
class coef_function3d(object):
    '''
    3-dimensional coefficient function. Logic remains the same. 
    '''
    
    def __init__(self, value, Dt, Dx, Dy, DxDy, DxDx, DyDy):
        self.value = value
        self.Dt    = Dt
        self.Dx    = Dx
        self.Dy    = Dy
        self.DxDy  = Dxdy
        self.DxDx  = Dxdx
        self.DyDy  = Dydy
        
    def value(self, t, x, y):
        return self.value(t, x, y)
    
    def dt(self, t, x, y):
        return self.Dt(t, x, y)
    
    def dx(self, t, x, y):
        return self.Dx(t, x, y)
    
    def dy(self, t, x, y):
        return self.Dy(t, x, y)
    
    def dxdy(self, t, x, y):
        return self.DxDy(t, x, y)
    
    def dxdx(self, t, x, y):
        return self.DxDx(t, x, y)
    
    def dydy(self, t, x, y):
        return self.DyDy(t, x, y)   
    
    
class mean_reversion(object):
    '''
    Class for correlation mean-reverted drift function.
    It depends on N+1 arguments: nu1, ... , nuN, rho. 
    '''
    def __init__(self, value, gradient, hessian):
        self.value    = value
        self.gradient = gradient
        self.hessian  = hessian
        
    def dx(self, NuVector, rho, coordinate):
        return self.gradient[coordinate](NuVector, rho)
    
    def dxdx(self, NuVector, rho, i, j):
        return self.gessian[i][j](NuVector, rho)

########################################################################################################################


# SDE PROCESSES ########################################################################################################

class ParticleProcess(object):
    '''
    This process describes the behaviour of the asset price. 
    
    The process can be described by an SDE:
    dS = drift_coef(t, S)dt + diffusion_coef(t, S, Nu)dWt, where Nu is the volatility process.
    
    NOT IMPLEMENTED YET
    
    PARAMS:
    S0:             the initial value of our diffusion. A number or a vector of numbers.
    drift_coef:     2d function, depends on time and process value. 
    diffusion_coef: 3d function, depends on time, process value and volatility value. 
    '''
    
    def __init__(self, S0, drift_coef, diffusion_coef):
        self.S0             = S0
        self.drift_coef     = drift_coef
        self.diffusion_coef = diffusion_coef
        

    

class VolatilityProcess(object):
    '''
    This process describes the behaviour of the asset price. 
    
    The process can be described by an SDE:
    dNu = drift_coef(t, Nu)dt + diffusion_coef(t, Nu)dBt, where Nu is the volatility process.
    
    NOT IMPLEMENTED YET
    
    PARAMS:
    Nu0:            the initial value of our diffusion. A number or a vector of numbers.
    drift_coef:     2d function, depends on time and volatility value. 
    diffusion_coef: 2d function, depends on time, and volatility value. 
    '''
    
    def __init__(self, Nu0, drift_coef, diffusion_coef):
        self.Nu0            = Nu0
        self.drift_coef     = drift_coef
        self.diffusion_coef = diffusion_coef

        
    

class CorrelationProcess(object):
    '''
    This process describes the behaviour of the asset price. 
    
    The process can be described by an SDE:
    dNu = drift_coef(t, Nu)dt + diffusion_coef(t, Nu)dBt, where Nu is the volatility process.
    
    NOT IMPLEMENTED YET
    
    PARAMS:
    rho0:           the initial value of our process. A number or a vector of numbers.
    mean_reversion: Psy(Nu1, ... , NuN) - rho_t. 
    diffusion_coef: 1d function, depends only on correlation value. 
    '''
    def __init__(self, rho0, alpha, mean_reversion, diffustion_coef):
        self.rho0           = rho0
        self.alpha          = alpha
        self.drift_coef     = self.alpha * mean_reversion
        self.diffusion_coef = diffusion_coef

########################################################################################################################



# STOCHASTIC CORRELATION MODEL #########################################################################################

class StochasticCorrModel(object):
    
    def __init__(Particles, Vols, Corr_Process, VolCorrMatrix, ParticleVolCorrMatrix, rho_skew):
        '''
        Attributes:
        Particles    -- the N-dimensional vector-process that describes asset prices. 
        Vols         -- the N-dimensional vector-process that descrives volatilities.
        Corr_Process -- the 1-d process that describes the correlation between brownian motions. 
        
        GeneralCorrMatrix  -- matrix of size 2N+1 x 2N+1 that contains all the correlations. 
        '''
        self.particles         = Particles
        self.vols              = vols
        self.corr_process      = Corr_Process
       
        self.N                 = len(Particles)
        
        self.VolCorrMatrix             = VolCorrMatrix
        self.ParticleVolCorrMatrix     = ParticleVolCorrMatrix
        
    def GeneralCorrMatrix(self, cur_rho):
        '''
        This method returns the current value of the General stochastic correlation matrix. 
        
        RETURNS: np.array of size 2N+1x2N+1. 
        '''
        return CalculateGeneralCorrMatrix(cur_rho, self.rho_skew, 
                                          self.ParticleVolCorrMatrix, self.VolCorrMatrix, self.N)


########################################################################################################################

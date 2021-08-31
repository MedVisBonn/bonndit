#%%cython --annotate
#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True,
# warn.unused_results=True

# These parameters control optimization of rank-1 approximations
ctypedef struct TijkRefineRank1:
    double eps_start
    double eps_impr
    double beta
    double gamma
    double sigma
    int maxTry
#    def __init__(self, eps_start=1e-10, eps_impr=1e-4, beta=0.3, gamma=0.9, sigma=0.5, maxTry=50):
#        # only do optimization if norm of deviatoric is larger than eps_start
#        self.eps_start = eps_start
#        # declare convergence if improvement is less than eps_impr times the norm of deviatoric (not residual)
#        self.eps_impr = eps_impr#
#
#        ## Parameters associated with Armijo stepsize control
#        # initial stepsize (divided by norm of deviatoric)
#        self.beta = beta
#        # stepsize reduction factor (0,1)
#        self.gamma = gamma
#        # corridor of values that lead to acceptance (0,1)
#        self.sigma = sigma
#        # number of stepsize reductions before giving up
#        self.maxTry = maxTry


ctypedef struct TijkRefineRank:
    double eps_res
    double eps_impr
    double pos
    TijkRefineRank1 rank1_parm



# These parameters control optimization of rank-k approximations
# class TijkRefineRankkParm:
#    def __init__(self, eps_res=1e-10, eps_impr=1e-4, pos=0):
#        # stop optimization if the residual is smaller than this
#        self.eps_res = eps_res
#        # declare convergence if tensor norm improved less than eps_impr times the original norm
#        self.eps_impr = eps_impr
#        # if non-zero, allow positive terms only
#        self.pos = pos
#        # used for rank1-optimization
#        #self.rank1_parm = TijkRefineRank1Parm()


# Holds information about (and functions needed for processing) a
# specific type of tensor
ctypedef struct TijkType:
    int order
    int dim
    int num

#    def __init__(self, name, order, dim, num):
#        self.name = name
#        self.order = order
#        self.dim = dim
#        self.num = num





cdef TijkRefineRank1 TijkRefineRank1Parm
cdef TijkRefineRank TijkRefineRankkParm
cdef int DIM


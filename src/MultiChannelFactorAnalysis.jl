module MultiChannelFactorAnalysis

import Base.+

export MultiChannelLayout, MultiChannelData, MultiChannelFactors, MultiChannelFactorLayout,  MCFMParams, MCFMHistory, MCFMFit
export fit, extract_factors, predict
export stack, unstack, extract_diagonal_blocks, extract_horizontal_blocks
export periodic_factors, classical_factors,indep_gauss_errors, simulate

using Statistics
using Distributions
using LinearAlgebra
using DataFrames

include("structs.jl")
include("utils.jl")
include("MCFM.jl")
include("MCFMSim.jl")



end # module

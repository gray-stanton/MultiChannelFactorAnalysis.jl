# Elementwise addition of two Datasets, assuming same size
function (+)(a::MultiChannelData, b::MultiChannelData)
    if a.channellayout.nobs_per_channel != b.channellayout.nobs_per_channel
        throw(ErrorException("Data observations per channel not equal"))
    elseif a.nsamples != b.nsamples
        throw(ErrorException("Number of samples not equal"))
    else
        vals = [[c1 .+ c2 for (c1, c2) in zip(samp1, samp2)] for (samp1, samp2) in zip(a.values, b.values)]
        out = MultiChannelData(vals, a.channellayout, a.nsamples)
        return out
    end
end


function eigenvalue_ratio_test(evs :: Array{Float64, 1}; threshold=5)
	# assume evs all real, sorted from largest to smallest, strictly positive
	# Could use random matrix theory to pick thresh.
	evratios = [min(ev1/ev2, typemax(Int64)) for (ev1, ev2) in zip(evs[1:(end-1)], evs[2:end])]
	rmax = maximum(evratios)
	if rmax < threshold
		facnum=0
	else
		facnum = argmax(evratios)
	end
	return facnum
end


function LTOrthog(B, blocksize)
    Blq = lq(B[1:blocksize, 1:blocksize])
    # Extract diagonal elems, sort into descending abs val, put back in diag mat.
    Lsigns = diagm(sign.(sort(diag(Blq.L), lt=(x, y) -> abs(y) < abs(x) )))
    Q = transpose(Blq.Q) * Lsigns  # eqn 11
    return Q
end

function extract_horizontal_blocks(mat, blocksizes; get_view=false)
    rbounds = vcat([0], cumsum(blocksizes))
    if get_view
        blocks = [view(mat, start:stop, :) for (start, stop) in zip(rbounds .+ 1, rbounds[2:end])]
    else
        blocks = [mat[start:stop, :] for (start, stop) in zip(rbounds .+ 1, rbounds[2:end])]
    end
    return blocks
end


function extract_diagonal_blocks(mat, rowblocksizes, colblocksizes; get_view=false)
    rbounds = vcat([0], cumsum(rowblocksizes))
    cbounds = vcat([0], cumsum(colblocksizes))
    if get_view
        blocks =  [view( mat, rstart:rstop, cstart:cstop)
            for (rstart, rstop, cstart, cstop) in zip(rbounds .+ 1, rbounds[2:end], cbounds .+ 1, cbounds[2:end])]
    else
        blocks = [mat[rstart:rstop, cstart:cstop]
            for (rstart, rstop, cstart, cstop) in zip(rbounds .+ 1, rbounds[2:end], cbounds .+ 1, cbounds[2:end])]
    end
    return blocks
end





# function extract_blocks(mat, Ls, pjs; diag=false)
#     rbounds = vcat([0], cumsum(Ls))
#     cbounds = vcat([0], cumsum(pjs))
#     if diag
#         # Pull out diagonal blocks
#         blocks = [mat[rstart:rstop, cstart:cstop]
#                     for (rstart, rstop, cstart, cstop) in zip(rbounds .+ 1, rbounds[2:end], cbounds .+ 1, cbounds[2:end])]
#     else
#         # Split rows of matrix into pieces.
#         blocks = [mat[start:stop, :] for (start, stop) in zip(rbounds .+ 1, rbounds[2:end])]
#     end
#     return blocks
# end

function flip_signs(mat)
    diag_signs = sign.(diag(mat))
    flip= mat * diagm(diag_signs)
    return flip
end

function blockdiag(blks; sparse=false)
    if sparse
        throw("unimplemented")
    else
        ncols = [size(blk)[2] for blk in blks]
        nrows = [size(blk)[1] for blk in blks]
        blkmat = zeros(sum(nrows), sum(ncols))
        colbounds = vcat([0], cumsum(ncols))
        rowbounds = vcat([0], cumsum(nrows))
        for (i, blk) in enumerate(blks)
            cstart, cstop = colbounds[i]+ 1, colbounds[i+1]
            rstart, rstop = rowbounds[i]+ 1, rowbounds[i+1]
            blkmat[rstart:rstop, cstart:cstop] = blk
        end
    end
    return blkmat
end


function stack(data :: MultiChannelData)
    X = hcat([vcat(samp...) for samp in data.values ]...)
    @assert size(X) == (sum(data.channellayout.nobs_per_channel), data.nsamples)
    return X
end

function unstack(X :: Matrix, channellayout::MultiChannelLayout; obs_ids = [], sample_ids = [])
    nsamples = size(X)[2]
    values = Array{Array{Array{Float64, 1}, 1}, 1}(undef, nsamples)
    for samp in 1:nsamples
        v = extract_horizontal_blocks(X[:, samp], channellayout.nobs_per_channel)
        values[samp] = [dropdims(vb; dims=2) for vb in v]
    end
    data = MultiChannelData(values, channellayout, nsamples; obs_ids, sample_ids)
    return data
end



#
# function stack_and_view(data)
#     N = length(data)
#     x1, xview1 = stack_and_view_single(data[1])
#     M = length(xview1)
#     L = sum([length(xc) for xc in x1])
#     X = Matrix(zeros((L, N))) # M x N, flat matrix
#     Xobs = repeat([xview1], N) # Obs-first, N x M x L_m
#     Xchan = [[Xobs[n][m] for n in 1:N] for m in 1:M ] # Chan first, M x N x L_m
#     X[:, 1] = x1
#     for n in 2:length(data)
#         #TODO: This doesn't accomplish the desired view, as view is to data
#         # underlying x, which is copied into matrix X. Need view into X itself,
#         # which requires rewriting this.
#         x, xobs = stack_and_view_single(data[n])
#         X[:, n] = x
#         Xobs[n] = xobs
#         for m in 1:M
#             Xchan[m][n] = xobs[m]
#         end
#     end
#     return X, Xobs, Xchan
# end
#
#
# # Probably need to separaete viewing and stacking code.
# function stack_and_view_single(xs)
#     # Combine ragged 3-dim array into matrix + view with struct.
#     lengs = [length(x) for x in xs]
#     bounds = vcat([0], cumsum(lengs))
#     x = vcat(xs...)
#     xview = [view(x, start:stop) for (start, stop) in zip(bounds .+ 1, bounds[2:end])]
#     return x, xview
# end

function data_fromlong(df; should_interpolate=true, max_missingpcent = 0.3 )
	#df must have 4 columns: "sample_id", "channel", "observation_id", "value"
	samp_ids = sort(unique(df.sample_id))
	channels = unique(df.channel)
	chanouts = Array{Array{Float64, 2}, 1}(undef, length(channels))
	nobs_per_channel = zeros(length(channels))
	k = 1
	local included_obs
	for chandf in groupby(df, :channel)
		obs_ids  = sort(unique(chandf.observation_id))
		included_rows = []
		included_obs = []
		chanout = zeros(length(obs_ids), length(samp_ids))
		for (j ,obs) in enumerate(obs_ids)
			obsdf = filter(:observation_id => id -> id == obs, chandf)
			sort!(obsdf, :sample_id)
			ts = obsdf.sample_id
			vals = obsdf.value
			outvals = zeros(length(samp_ids))
			if length(ts) != length(samp_ids)
				if should_interpolate
					if length(ts) <= (1-max_missingpcent)*length(samp_ids)
						# too many missing, should not be included in model.
						print("$obs has too many missing values, removed.\n")
						# Some misingness pattern problem though. What if different
						# numbers of obs for same obs_id are missing in different channels?
						continue
					else
						int = LinearInterpolation(ts, vals, extrapolation_bc=Line())
						interpolated_values = int(samp_ids)
						# outvals is full-length vector, where the observed values
						# are present, and missing values filled in
						for (i, t) in enumerate(samp_ids)
							outvals[i] = t in ts ? vals[findfirst(ts .== t)] : interpolated_values[i]
						end
					end
				else
					throw(ErrorException("Missing observations and not interpolating!"))
				end
			else
				outvals = vals
			end
			push!(included_obs, obs)
			push!(included_rows, j)
			chanout[j, :] = outvals
		end
		nobs_per_channel[k] = length(included_obs)
		chanouts[k] = chanout[included_rows, :]
		k += 1
	end
	stacked_X = cat(chanouts...; dims=1) :: Array{Float64, 2}
	convert(Array{Float64, 2}, stacked_X)
	channellayout = MultiChannelLayout(nobs_per_channel)
	data = unstack(stacked_X, channellayout; obs_ids = included_obs, sample_ids = samp_ids)
	return data
end


function to_long(params::MCFMParams; obs_ids = Int[])
	nchannels = params.channellayout.nchannels
	h_longs = Array{DataFrame, 1}(undef, nchannels)
	g_longs = Array{DataFrame, 1}(undef, nchannels)
	for (c, Hchan) in enumerate(params.H_by_channel)
		df = DataFrame(Hchan)
		rename!(df, ["common" * string(i) for i in 1:ncol(df)])
		df.obs = length(obs_ids) == nrow(df) ? obs_ids : axes(df, 1)
		df_long = DataFrames.stack(df, Not(:obs))
		df_long[!, :channel] .= c
		rename!(df_long, :variable => "factor")
		h_longs[c] = df_long
	end
	for (c, Gchan) in enumerate(params.G_by_channel)
		df = DataFrame(Gchan)
		rename!(df, ["channel$(c)_unique" * string(i) for i in 1:ncol(df)])
		df.obs = length(obs_ids) == nrow(df) ? obs_ids : axes(df, 1)
		df_long = DataFrames.stack(df, Not(:obs))
		df_long[!, :channel] .= c
		rename!(df_long, :variable => "factor")
		g_longs[c] = df_long
	end

	Hlong = vcat(h_longs...)
	Glong = vcat(g_longs...)
	out = vcat(Hlong, Glong)
	return out
end



function to_long(factors :: MultiChannelFactors; sample_ids = Int[])
	factor_names = ["common" * string(i) for i in  1:factors.factorlayout.nchannelshared]
	for (c, nc) in enumerate(factors.factorlayout.nspecific_per_channel)
		factor_names = vcat(factor_names, ["channel$(c)_unique" * string(i) for i in 1:nc])
	end
	df = DataFrame(transpose(factors.factors))
	rename!(df, factor_names)
	df.sample_id = length(sample_ids) == nrow(df) ? sample_ids : axes(df, 1)
	df_long = DataFrames.stack(df, Not(:sample_id))
	rename!(df_long, :variable => "factor")
	return df_long
end


function to_long(data :: MultiChannelData; obs_ids = Int[], sample_ids = Int[])
	X = MultiChannelFactorAnalysis.stack(data)
	nchannels = data.channellayout.nchannels
	chanblocks = extract_horizontal_blocks(X, data.channellayout.nobs_per_channel)
	Xlongs = Array{DataFrame, 1}(undef, nchannels)
	for (c, blk) in enumerate(chanblocks)
		df = DataFrame(transpose(blk))
		df.sample_id = length(sample_ids) == nrow(df) ? sample_ids : axes(df, 1)
		df_long = DataFrames.stack(df, Not(:sample_id))
		df_long[!, :channel] .= c
		rename!(df_long, :variable => "obs_str")
		# Drop the "x" in front of obs_id (stored as Categorical by default.)
		fixed_obs = [parse(Int, levels(x)[1][2:end]) for x in df_long[:obs_str]]
		df_long[!, :obs] = fixed_obs
		if length(obs_ids) == ncol(df) -1 # added sample_id
			# index into the given ids (assuming in order).
			df_long[!, :obs] = obs_ids[df_long[!, :obs]]
		end
		select!(df_long, Not(:obs_str))
		Xlongs[c] = df_long
	end
	out = vcat(Xlongs...)
	return out
end








function parse_df(df)
    #df must have 3 columns: "sample", "channel", "observation"
    # must be sorted appropriately
    nsamples = length(unique(df.sample))
    nchannel = length(unique(df.channel))

    values = Array{Array{Array{Float64, 1}, 1}, 1}(undef, nsamples)
    n = 1
    for sampdf in groupby(df, :sample)
        sample_dat = Array{Array{Float64, 1}, 1}(undef, nchannel)
        i = 1
        for chandf in groupby(sampdf, :channel) # Potential sorting bug here
            obs = copy(chandf.observation)
            sample_dat[i] = obs
            i += 1
        end
        values[n] = sample_dat
        n +=1
    end
    channellayout = MultiChannelLayout([size(c)[1] for c in values[1]])
    data = MultiChannelData(values, channellayout, nsamples)
    return data
end


"""
	varimax(A; gamma = 1.0, minit = 20, maxit = 1000, reltol = 1e-12)
VARIMAX perform varimax (or quartimax, equamax, parsimax) rotation to the column vectors of the input matrix.
# Input Arguments
- `A::Matrix{Float64}`: input matrix, whose column vectors are to be rotated. d, m = size(A).
- `gamma::Float64`: default is 1. gamma = 0, 1, m/2, and d(m - 1)/(d + m - 2), corresponding to quartimax, varimax, equamax, and parsimax.
- `minit::Int`: default is 20. Minimum number of iterations, in case of the stopping criteria fails initially.
- `maxit::Int`: default is 1000. Maximum number of iterations.
- `reltol::Float64`: default is 1e-12. Relative tolerance for stopping criteria.
# Output Argument
- `B::Matrix{Float64}`: output matrix, whose columns are already been rotated.
Implemented by Haotian Li, Aug. 20, 2019
"""
function varimax(A; gamma = 1.0, minit = 20, maxit = 1000, reltol = 1e-12)
	# Get the sizes of input matrix
	d, m = size(A)

	# If there is only one vector, then do nothing.
	if m == 1
		return A
	end

	# Warm up step: start with a good initial orthogonal matrix T by SVD and QR
	T = Matrix{Float64}(I, m, m)
	B = A * T
	L,_,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
	T = L * M'
	if norm(T-Matrix{Float64}(I, m, m)) < reltol
		T,_ = qr(randn(m,m)).Q
		B = A * T
	end

	# Iteration step: get better T to maximize the objective (as described in Factor Analysis book)
	D = 0
	for k in 1:maxit
		Dold = D
		L,s,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
		T = L * M'
		D = sum(s)
		B = A * T
		if (abs(D - Dold)/D < reltol) && k >= minit
			break
		end
	end

	# Adjust the sign of each rotated vector such that the maximum absolute value is positive.
	for i in 1:m
		if abs(maximum(B[:,i])) < abs(minimum(B[:,i]))
			B[:,i] .= - B[:,i]
		end
	end

	return B
end

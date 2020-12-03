
function LTOrthog(B, blocksize)
    Blq = lq(B[1:blocksize, 1:blocksize])
    # Extract diagonal elems, sort into descending abs val, put back in diag mat.
    Lsigns = diagm(sign.(sort(diag(Blq.L), lt=(x, y) -> abs(y) < abs(x) )))
    Q = transpose(Blq.Q) * Lsigns  # eqn 11
    return Q
end


function extract_blocks(mat, Ls, pjs; diag=false)
    rbounds = vcat([0], cumsum(Ls))
    cbounds = vcat([0], cumsum(pjs))
    if diag
        # Pull out diagonal blocks
        blocks = [mat[rstart:rstop, cstart:cstop]
                    for (rstart, rstop, cstart, cstop) in zip(rbounds .+ 1, rbounds[2:end], cbounds .+ 1, cbounds[2:end])]
    else
        # Split rows of matrix into pieces.
        blocks = [mat[start:stop, :] for (start, stop) in zip(rbounds .+ 1, rbounds[2:end])]
    end
    return blocks
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



function stack_and_view(data)
    N = length(data)
    x1, xview1 = stack_and_view_single(data[1])
    M = length(xview1)
    L = sum([length(xc) for xc in x1])
    X = Matrix(zeros((L, N))) # M x N, flat matrix
    Xobs = repeat([xview1], N) # Obs-first, N x M x L_m
    Xchan = [[Xobs[n][m] for n in 1:N] for m in 1:M ] # Chan first, M x N x L_m
    X[:, 1] = x1
    for n in 2:length(data)
        #TODO: This doesn't accomplish the desired view, as view is to data
        # underlying x, which is copied into matrix X. Need view into X itself,
        # which requires rewriting this.
        x, xobs = stack_and_view_single(data[n])
        X[:, n] = x
        Xobs[n] = xobs
        for m in 1:M
            Xchan[m][n] = xobs[m]
        end
    end
    return X, Xobs, Xchan
end


# Probably need to separaete viewing and stacking code.
function stack_and_view_single(xs)
    # Combine ragged 3-dim array into matrix + view with struct.
    lengs = [length(x) for x in xs]
    bounds = vcat([0], cumsum(lengs))
    x = vcat(xs...)
    xview = [view(x, start:stop) for (start, stop) in zip(bounds .+ 1, bounds[2:end])]
    return x, xview
end

function parse_df(df)
    #df must have 3 columns: "sample", "channel", "observation"
    # must be sorted appropriately
    nsamples = length(unique(df.sample))
    nchannel = length(unique(df.channel))

    data = Array{Array{Array{Float64, 1}, 1}, 1}(undef, nsamples)
    n = 1
    for sampdf in groupby(df, :sample)
        sample_dat = Array{Array{Float64, 1}, 1}(undef, nchannel)
        i = 1
        for chandf in groupby(sampdf, :channel) # Potential sorting bug here
            obs = copy(chandf.observation)
            sample_dat[i] = obs
            i += 1
        end
        data[n] = sample_dat
        n +=1
    end
    return data
end

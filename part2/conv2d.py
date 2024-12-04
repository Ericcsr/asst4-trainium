import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_out_per_tile = min(out_channels, nl.tile_size.gemm_stationary_fmax) 
    n_tiles_c_out = out_channels // c_out_per_tile

    c_in_per_tile = min(in_channels, nl.tile_size.pmax)
    n_tiles_c_in = in_channels // c_in_per_tile

    # Handle one line of output at a time
    total_out = out_height * out_width
    tile_size = pool_size * out_width if total_out >= nl.tile_size.gemm_moving_fmax else total_out
    n_tiles_hw = out_height * out_width // tile_size
    n_vert_pools = tile_size // (pool_size * out_width)
    tile_height = tile_size // out_width

    # Process the images in batches
    # out of memory during compilation...?
    # TODO: Should swap order of loops to avoid
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]
        W_cache = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_per_tile), in_channels, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
        for tile_id in nl.affine_range(n_tiles_c_out):
            for in_id in nl.affine_range(in_channels):
                W_cache[tile_id, :,in_id,:,:] = nl.load(W[tile_id*c_out_per_tile:(tile_id+1)*c_out_per_tile,in_id,:,:])
        for n in nl.affine_range(n_tiles_c_out):
            bias_tile = nl.ndarray((c_out_per_tile,1,1), dtype=bias.dtype, buffer=nl.sbuf)
            bias_tile[...] = nl.load(bias[n * c_out_per_tile:(n + 1) * c_out_per_tile])
            broadcasted_bias = bias_tile.broadcast_to((c_out_per_tile, n_vert_pools, out_pool_width))
            for m in nl.affine_range(n_tiles_hw):
                conv_result = nl.zeros((nl.par_dim(c_out_per_tile), tile_height * out_width), dtype=X_out.dtype, buffer=nl.sbuf)
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        # partial sum in psum  
                        res_psum = nl.zeros((c_out_per_tile, tile_height * out_width), nl.float32, buffer=nl.psum)
                        for k in nl.affine_range(n_tiles_c_in):
                            Wt_tile = nl.ndarray((c_out_per_tile, c_in_per_tile), dtype=W.dtype, buffer=nl.sbuf)
                            #for l in nl.affine_range(c_out_per_tile):
                            Wt_tile[...] = nl.copy(W_cache[n, :, k * c_in_per_tile:(k + 1) * c_in_per_tile, i, j], dtype=W.dtype)

                            # Should make X_tile preloaded as well
                            X_tile = nl.ndarray((c_in_per_tile, tile_height * out_width), dtype=X.dtype, buffer=nl.sbuf)
                            for h in nl.affine_range(tile_height):
                                X_tile[:, h*out_width:(h+1)*out_width] = nl.load(X[b, k * c_in_per_tile:(k + 1) * c_in_per_tile, m * tile_height + h + i, j:j+out_width])
                                
                            #res_psum += nisa.nc_matmul(Wt_tile[...], X_tile[...],is_transpose=True) # directly write to psum
                            res_psum += nl.matmul(Wt_tile[...], X_tile[...], transpose_x=False)
                        conv_result[...] = nl.loop_reduce(res_psum, op=np.add, loop_indices=[i,j], dtype=X_out.dtype) # directly transfer sbuf

                i_0 = nl.arange(c_out_per_tile)[:, None, None, None, None]
                i_1 = nl.arange(n_vert_pools)[None, :, None, None, None]
                i_2 = nl.arange(out_pool_width)[None, None, :, None, None]
                i_3 = nl.arange(pool_size)[None, None, None, :, None]
                i_4 = nl.arange(pool_size)[None, None, None, None, :]
                # Should try to load as a 3D tensor first
                out_tile = nisa.tensor_reduce(np.max,conv_result[i_0, (i_1 * pool_size + i_3) * out_width + i_2 * pool_size + i_4], axis=[3, 4])
                out_tile += broadcasted_bias
                out_tile_ = nl.copy(out_tile, dtype=X_out.dtype)
                #breakpoint()
                nl.store(X_out[b, n * c_out_per_tile:(n + 1) * c_out_per_tile, m * n_vert_pools:(m + 1) * n_vert_pools, :], value=out_tile_)
                #breakpoint()
                # without maxpooling direct indexing
                # i_0 = nl.arange(c_out_per_tile)[:, None, None]
                # i_1 = nl.arange(n_vert_pools)[None, :, None]
                # i_2 = nl.arange(out_pool_width)[None, None, :]
                # out_tile = conv_result[i_0, i_1 * out_pool_width + i_2]
                # out_tile += broadcasted_bias
                # nl.store(X_out[b, n * c_out_per_tile:(n + 1) * c_out_per_tile, m * n_vert_pools:(m + 1) * n_vert_pools, :], value=out_tile)

    return X_out
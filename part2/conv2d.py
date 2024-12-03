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

    # load X and W and bias to memory
    X_ = nl.ndarray(
        shape=(batch_size, nl.par_dim(in_channels), input_height, input_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    X_[...] = nl.load(X, dtype=X.dtype)

    # W_ = nl.ndarray(
    #     shape = (out_channels, nl.par_dim(in_channels), filter_height, filter_width),
    #     dtype=X.dtype,
    #     buffer=nl.hbm
    # )
    # for i in nl.affine_range(out_channels):
    #     W_[i] = nl.load(W[i], dtype=X.dtype)
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
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]
        # convolution first 
        for n in nl.affine_range(n_tiles_c_out):
            bias_tile = nl.ndarray((c_out_per_tile,1,1), dtype=bias.dtype, buffer=nl.sbuf)
            bias_tile[...] = nl.load(bias[n * c_out_per_tile:(n + 1) * c_out_per_tile])
            broadcasted_bias = bias_tile.broadcast_to((c_out_per_tile, n_vert_pools, out_pool_width))
            for m in nl.affine_range(n_tiles_hw):
                conv_result = nl.zeros((c_out_per_tile, tile_height * out_width), dtype=X_out.dtype, buffer=nl.sbuf)
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        # partial sum in psum  
                        res_psum = nl.zeros((c_out_per_tile, tile_height * out_width), nl.float32, buffer=nl.psum)
                        for k in nl.affine_range(n_tiles_c_in):
                            Wt_tile = nl.ndarray((c_in_per_tile, c_out_per_tile), dtype=W.dtype, buffer=nl.sbuf)
                            for l in nl.affine_range(c_out_per_tile):
                                Wt_tile[:, l] = nl.load(W[n * c_out_per_tile + l, k * c_in_per_tile:(k + 1) * c_in_per_tile, i, j])

                            X_tile = nl.ndarray((c_in_per_tile, tile_height * out_width), dtype=X.dtype, buffer=nl.sbuf)
                            for h in nl.affine_range(tile_height):
                                X_tile[:, h*out_width:(h+1)*out_width] = nl.load(X_[b, k * c_in_per_tile:(k + 1) * c_in_per_tile, m * tile_height + h + i, j:j+out_width])
                                
                            res_psum += nisa.nc_matmul(Wt_tile[...], X_tile[...]) # directly write to psum
                        conv_result[...] = nl.loop_reduce(res_psum, op=np.add, loop_indices=[i,j], dtype=X_out.dtype) # directly transfer sbuf

                i_0 = nl.arange(c_out_per_tile)[:, None, None, None, None]
                i_1 = nl.arange(n_vert_pools)[None, :, None, None, None]
                i_2 = nl.arange(out_pool_width)[None, None, :, None, None]
                i_3 = nl.arange(pool_size)[None, None, None, :, None]
                i_4 = nl.arange(pool_size)[None, None, None, None, :]
                #conv_result = nl.copy(conv_result, dtype=X_out.dtype)
                out_tile = nl.max(conv_result[i_0, (i_1 * pool_size + i_3) * out_width + i_2 * pool_size + i_4], axis=[3, 4])
                out_tile += broadcasted_bias
                nl.store(X_out[b, n * c_out_per_tile:(n + 1) * c_out_per_tile, m * n_vert_pools:(m + 1) * n_vert_pools, :], value=out_tile)
    return X_out

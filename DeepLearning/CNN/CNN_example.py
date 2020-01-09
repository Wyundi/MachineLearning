def conv_forward_naive(x, w, b, conv_param):
    """ A naive implementation of the forward pass for a convolutional layer. 
    The input consists of N data points, each with C channels, height H and width W. 
    We convolve each input with F different filters, where each filter spans all C channels 
    and has height HH and width HH. Input: - x: Input data of shape (N, C, H, W) - w: Filter 
    weights of shape (F, C, HH, WW) - b: Biases, of shape (F,) - conv_param: A dictionary with 
    the following keys: - 'stride': The number of pixels between adjacent receptive fields in 
    the horizontal and vertical directions. - 'pad': The number of pixels that will be used to 
    zero-pad the input. Returns a tuple of: - out: Output data, of shape (N, F, H', W') where 
    H' and W' are given by 
    H' = 1 + (H + 2 * pad - HH) / stride 
    W' = 1 + (W + 2 * pad - WW) / stride - cache: (x, w, b, conv_param) """
    out = None
    N, C, H, W = x.shape
    # N data points, each with C channels, height H and width W.
    F, C, HH,WW= w.shape
    # F different filters, where each filter spans all C channels and has height HH and width HH.
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    X = np.pad(x, ((0,0), (0, 0), (pad, pad),(pad, pad)), 'constant')
    
    Hn = 1 + int((H + 2 * pad - HH) / stride)
    Wn = 1 + int((W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, Hn, Wn))
    for n in range(N):
        for m in range(F):
            for i in range(Hn):
                for j in range(Wn):
                    data = X[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW].reshape(1, -1)
                    filt = w[m].reshape(-1, 1)
                    out[n, m, i, j] = data.dot(filt) + b[m]
    cache = (x, w, b, conv_param)
    return out, cache

def conv_forward_im2col(x, w, b, conv_param):
    """ A fast implementation of the forward pass for a convolutional layer based on im2col and col2im. """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache









def conv_backward_naive(dout, cache):
    """ A naive implementation of the backward pass for a convolutional layer. 
    Inputs: - dout: Upstream derivatives. - cache: A tuple of (x, w, b, conv_param) 
    as in conv_forward_naive Returns a tuple of: - dx: Gradient with respect to 
    x - dw: Gradient with respect to w - db: Gradient with respect to b """
    x, w, b, conv_param = cache
    N, F, Hn, Wn = dout.shape
    N, C, H, W = x.shape
    F, C, HH,WW= w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    dw = np.zeros_like(w)
    X = np.pad(x, ((0,0), (0, 0), (pad, pad),(pad, pad)), 'constant')
    dX = np.zeros_like(X)
    for n in range(N):
        for m in range(F):
            for i in range(Hn):
                for j in range(Wn):
                    dX[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[m] *  dout[n, m, i, j]
                    dw[m] += X[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] *  dout[n, m, i, j]
    db = np.sum(dout, axis=(0, 2, 3))
    dx = dX[:, :, pad:-pad, pad:-pad]
    return dx, dw, db

def conv_backward_im2col(dout, cache):
    """ A fast implementation of the backward pass for a convolutional layer based on im2col and col2im. """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
    dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                       filter_height, filter_width, pad, stride)

    return dx, dw, db


def im2col_cython(x, WW, HH, pad, stride):
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                  shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)
    return x_cols
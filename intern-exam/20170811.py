#Conveolution layer Imprement
import sys, os
sys.path.append(os.parder)
from common.util import im2col

x1 = np.random.rand(1, 3,7,7)
col2 = im2col(x1, 5,5,stride=1, pad=0)
print(col1.shape)
x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2, 5, 5,stride=1, pad=0)
print(col2.shape)

class Convolution
  def __init__(self, w, b, stride=1, pad=0):
    self.w =w
    self.b =b
    self.stride = stride
    self.pad = pad
  def forward(self, x):
    FN, C, FH, FW = self.W.shape
    N, C, H, W = x.shape
    out_h = int(1+(H + 2*self.pd - FH) / self.stride)
    out_w = int(1+(W+2*self.pad-FW)/self.stride)

    col = im2col(x, FH, FW, self.stride, self.pad)
    col_w = self.W.reshape(FN, -1).T
    out= np.dot(col, col_w) + self.b

    out = out.reshape(N,out_h, out_w, -1).transpose(0,3,1,2)

    return out

    class Pooling:
      def __init__ (self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad = pad

      def forward(self, x):
        N,C,H,W = x.shape
        out_h = init(1 + (H + self.pool_h) / self.stride)

        col = im2col(x, self.pool?h, self/pool_h*)
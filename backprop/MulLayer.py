class MulLayer.py
def __init__(self):
  self.x = None
  self.y = None

def forward(self, x, y):
  self.x = x
  self.y = y
  out = x * y

  return out

def backward(self, dout):
  sx = dout * self.y
  dy = dout * self.x

  return dx, dy
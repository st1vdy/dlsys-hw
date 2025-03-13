import os
import sys

sys.path.append("/content/needle/python")
os.environ['PYTHONPATH'] = '/content/needle/python:/env/python'
from needle import backend_ndarray as nd

x = nd.NDArray([[1, 2, 3], [4, 5, 6]], device=nd.cuda())
print(x, x.device)
y = nd.NDArray.make(
    shape=(3, 2),
    strides=(1, 3),
    offset=0,
    handle=x._handle,
    device=x.device,
)
print(y, y.device)

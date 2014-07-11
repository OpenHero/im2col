zhao.kaiyong@gmail.com
kyzhao@comp.hkbu.edu.hk
http://www.comp.hkbu.edu.hk/~kyzhao/
http://blog.csdn.net/openhero

Based on caffe im2col.
Merge the loop into one kernel.
On GTX640:
image 256*256 with 3 channels
batch size is 128
The time show as below:
caffe is 106.883766ms
bu_im2col is 22.095470ms
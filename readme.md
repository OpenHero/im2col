* 	created:	2014/07/11
	created:	11:7:2014   16:22
	author:		Zhao Kaiyong
	email: zhao.kaiyong(at)gmail.com, kyzhao(at)comp.hkbu.edu.hk
	website: http://www.comp.hkbu.edu.hk/~kyzhao/ , http://blog.csdn.net/openhero

* Based on caffe im2col.
Merge the loop into one kernel.
On GTX640:
image 256*256 with 3 channels
batch size is 128
The time show as below:
caffe is 106.883766ms
bu_im2col is 22.095470ms
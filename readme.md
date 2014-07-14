###  Based on caffe im2col. Merge the loop into one kernel.

On GTX640:

### The time show as below:
* 	image 
	const int height = 256;
	const int width = 256;
	const int channels = 3;
	const int batch_size = 32;//128;
	const int ksize = 5; // 5-11
	const int pad = 2; // 0-2
	const int stride = 1; // 1
	const int num_kernels = 64;

--------------------------------------------------
* author:		Zhao Kaiyong
* email: zhao.kaiyong(at)gmail.com, kyzhao(at)comp.hkbu.edu.hk
* website: http://www.comp.hkbu.edu.hk/~kyzhao/ , http://blog.csdn.net/openhero
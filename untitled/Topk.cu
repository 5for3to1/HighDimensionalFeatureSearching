#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdint.h>
#include <stdio.h>
#include<iostream>
#include<fstream>
#include<ctime>
#include<math.h>
#include<string>

using namespace std;

typedef unsigned char BYTE;
typedef float Real;

//存储最终选择出的相似度与图片路径
struct PathSimilarity
{
	string path;
	float similarity;
};

//存储海明距离和索引的结构体
struct IndexHamming
{
	int index;
	unsigned short hammingDis;
};

//存储余弦距离和索引的结构体
struct IndexCosine
{
	int index;
	float cosineDis;
};

//存储哈希码的结构体
struct HashCode
{
	unsigned char codeTtem[64];
};

#define TOTAL_DATA 256										//1000*1000
#define FEAT 512											//特征向量的位数

#define CHOOSEN_DATA 32										//海明距离筛选出的数量
#define face_num 5											//最终选择出人脸的数量



/**************************************************************/
/*****************工程所在文件夹的相对路径********************/
string ROOT_PATH = "G:/MyQt/";								//人脸图片和工程所在的路径


Real * real = new float[TOTAL_DATA*FEAT];					//存储人脸库实值
HashCode * hashcode = new HashCode[TOTAL_DATA];				//TOTAL_DATA 个HashCode结构体 的哈希码

Real * test_r = new float[FEAT];							//待测试的实值
HashCode * test_hc = new HashCode;

HashCode * g_hashcode;										//显存中存储人脸库哈希码


//计算P与Q的海明距离
__device__ unsigned short match(BYTE P, BYTE Q)
{
	unsigned short count = 0;
	BYTE result = P^Q;
	while (result)
	{
		result &= result - 1;
		count++;
	}
	return count;
	//return __popcnt16(unsigned short(P^Q));
}

__device__ unsigned short gmatch(BYTE*P, BYTE*Q, int codelb) {
	switch (codelb) {
	case 4: // 32 bit
		return __popc(*(BYTE*)P ^ *(BYTE*)Q);
	case 8: // 64 bit
		return __popcll(((unsigned long long*)P)[0] ^ ((unsigned long long*)Q)[0]);
	case 16: // 128 bit
		return __popcll(((unsigned long long*)P)[0] ^ ((unsigned long long*)Q)[0]) \
			+ __popcll(((unsigned long long*)P)[1] ^ ((unsigned long long*)Q)[1]);
	case 32: // 256 bit
		return __popcll(((unsigned long*)P)[0] ^ ((unsigned long *)Q)[0]) \
			+ __popcll(((unsigned long *)P)[1] ^ ((unsigned long *)Q)[1]) \
			+ __popcll(((unsigned long *)P)[2] ^ ((unsigned long *)Q)[2]) \
			+ __popcll(((unsigned long *)P)[3] ^ ((unsigned long *)Q)[3]);
	case 64: // 512 bit
		return __popcll(((unsigned long *)P)[0] ^ ((unsigned long *)Q)[0]) \
			+ __popcll(((unsigned long *)P)[1] ^ ((unsigned long *)Q)[1]) \
			+ __popcll(((unsigned long *)P)[2] ^ ((unsigned long *)Q)[2]) \
			+ __popcll(((unsigned long *)P)[3] ^ ((unsigned long *)Q)[3]) \
			+ __popcll(((unsigned long *)P)[4] ^ ((unsigned long *)Q)[4]) \
			+ __popcll(((unsigned long *)P)[5] ^ ((unsigned long *)Q)[5]) \
			+ __popcll(((unsigned long *)P)[6] ^ ((unsigned long *)Q)[6]) \
			+ __popcll(((unsigned long *)P)[7] ^ ((unsigned long *)Q)[7]);
	default:
		break;
	}
}

__global__ void calc_HammingDis_Kernel(HashCode * hashcode, HashCode * test_hc, IndexHamming * indexHamming)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < TOTAL_DATA )
	{
		//indexHamming[tid].hammingDis = 0;
		indexHamming[tid].index = tid;
		/*
		for (int i = 0; i < FEAT / 8; i++)
		{
		indexHamming[tid].hammingDis += match(hashcode[tid].codeTtem[i], test_hc->codeTtem[i]);
		}*/
		indexHamming[tid].hammingDis = gmatch(hashcode[tid].codeTtem, test_hc->codeTtem, FEAT / 8);
		__syncthreads();
	}
}

//len个数据，第i个数据与第 len-i 个数据进行比较
__global__ void bitonic_merge_Kernel(IndexHamming * indexHamming, int len)
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;			//当前线程号
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / 2)
	{
		int offset = index % (len / 2);								/*区块内偏移*/
		int base = index / (len / 2);								/*所在的区块号*/

		int source = base*len + offset;								/*线程对应的数据的下标*/
		int target = base*len + len - offset - 1;

		IndexHamming temp;													//用于交换的中间变量
		if (indexHamming[source].hammingDis >  indexHamming[target].hammingDis)
		{
			temp = indexHamming[source];
			indexHamming[source] = indexHamming[target];
			indexHamming[target] = temp;
		}
		__syncthreads();
	}
}

//len个数据，第i个数据与第 len-i 个数据进行比较
__global__ void head_tail_sort(IndexHamming * indexHamming, int stride)
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;				//当前线程号
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / stride)
	{
		int base = index / CHOOSEN_DATA * stride;					/*所在的区块号*/
		int offset = index % CHOOSEN_DATA;							/*区块内偏移*/

		int source = base*CHOOSEN_DATA + offset;					/*线程对应的数据的下标*/
		int target = (base + stride / 2)*CHOOSEN_DATA + CHOOSEN_DATA - offset - 1;

		//IndexHamming temp;													//用于交换的中间变量
		if (indexHamming[source].hammingDis >  indexHamming[target].hammingDis)
		{
			//temp = indexHamming[source];
			indexHamming[source] = indexHamming[target];
			//indexHamming[target] = temp;
		}
		__syncthreads();
	}
}

__global__ void head_tail_sort_single(IndexHamming * indexHamming, int stride)
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;				//当前线程号
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index<TOTAL_DATA / (CHOOSEN_DATA*stride))
	{
		int base = index*CHOOSEN_DATA*stride;
		for (int i = 0; i < CHOOSEN_DATA; i++)
		{
			if (indexHamming[base + CHOOSEN_DATA - 1 - i].hammingDis>indexHamming[base + CHOOSEN_DATA*stride / 2 + i].hammingDis)
			{
				indexHamming[base + CHOOSEN_DATA - 1 - i] = indexHamming[base + CHOOSEN_DATA*stride / 2 + i];
			}
			else
			{
				break;
			}
		}
	}
}

//将CHOOSEN_DATA个数据排成有序的
__global__ void head_choosen_sort(IndexHamming * indexHamming, int stride)		//len=CHOOSEN_DATA/2;
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;			//当前线程号
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / (stride * 2))
	{
		int base_const = index / (CHOOSEN_DATA / 2);					/*所在的区块号*/
		int offset_const = index % (CHOOSEN_DATA / 2);

		IndexHamming temp;										//用于交换的中间变量

		for (int len = CHOOSEN_DATA >> 1; len >= 1; len >>= 1)
		{
			int base = offset_const / len;									/*所在的区块号*/
			int offset = offset_const % len;								/*区块内偏移*/

			int source = base * len * 2 + offset + CHOOSEN_DATA * stride * base_const;		/*线程对应的数据的下标*/
			int target = source + len;

			if (indexHamming[source].hammingDis >  indexHamming[target].hammingDis)
			{
				temp = indexHamming[source];
				indexHamming[source] = indexHamming[target];
				indexHamming[target] = temp;
			}
		}
		__syncthreads();
	}
}

//len个数据，第i个数据与第 i+len/2 个数据进行比较
__global__ void bitonic_half_cleaner_Kernel(IndexHamming * indexHamming, int len)
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;			//当前线程号
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / 2)
	{
		int offset = index % (len / 2);								/*区块内偏移*/
		int base = index / (len / 2);								/*所在的区块号*/

		index = base*len + offset;									/*线程对应的数据的下标*/
		int target = index + len / 2;

		IndexHamming temp;											//用于交换的中间变量
		if (indexHamming[index].hammingDis >  indexHamming[target].hammingDis)
		{
			temp = indexHamming[index];
			indexHamming[index] = indexHamming[target];
			indexHamming[target] = temp;
		}
		__syncthreads();
	}
}


/*******************************************主机函数*****************************************/

void ProduceRandom(Real * real)					//伪随机产生 TOTAL_DATA*FEAT 个float
{
	for (int i = 0; i < TOTAL_DATA*FEAT; i++)
	{
		real[i] = rand() % 20 - 10;				//产生-10到10之间的随机数
	}
}

void ProduceRandom_test(Real * test_r)				//伪随机产生一个FEAT位的float数组
{
	for (int i = 0; i < FEAT; i++)
	{
		test_r[i] = rand() % 20 - 10;					//产生-10到10之间的随机数
	}
}


//专用于将百万个128位float的实值 编码 成百万个16个Byte的哈希码
void HashCoding(Real * real, HashCode * hashcode)
{
	for (int i = 0; i < TOTAL_DATA; i++)
	{

		for (int j = 0; j <FEAT; j++)
		{
			if (j % 8 == 0)												//每处理8个float对一个哈希码BYTE初始化
			{
				hashcode[i].codeTtem[j / 8] = 0;
			}
			if (real[j + i*FEAT] > 0)
			{
				hashcode[i].codeTtem[j / 8] += pow(2, 7 - (j % 8));		//将前8个float编码成一个Byte
			}
		}
	}
}

void HashCoding_test(Real * test_r, HashCode * test_hc)
{
	for (int i = 0; i <FEAT; i++)
	{
		if (i % 8 == 0)												//每处理8个float对一个哈希码初始化一次
		{
			test_hc->codeTtem[i / 8] = 0;
		}
		if (test_r[i]>0)
		{
			test_hc->codeTtem[i / 8] += pow(2, 7 - (i % 8));		//将前8个float编码成一个Byte
		}
	}
}

//读取lib中特征向量
void readFeature(string fileName, Real * real)
{

	ifstream in(fileName);				//文件流打开文件
	if (!in.is_open())
	{
		cout << "文件打开失败！" << endl;
	}
	else
	{
		int count = 0;
		while (!in.eof())
		{
			in >> real[count];
			count++;
		}
		//cout << "特征数：" << count/FEAT << endl;
	}
}

//读取test中特征向量
void readTestFeature(string fileName, Real * test_r, int index)
{
	ifstream in(fileName);				//文件流打开文件
	if (!in.is_open())
	{
		cout << "文件打开失败！" << endl;
	}
	else
	{
		int count = 0;
		float tem;
		while (!in.eof() && count < (index + 1)*FEAT)
		{
			in >> tem;
			if (count >= index*FEAT)
			{
				test_r[count - index*FEAT] = tem;
			}
			count++;
		}
	}
}


//读取人脸库的特征向量，传入显存中
extern "C"
void red_face_lib()
{
	readFeature(ROOT_PATH + "face/lib_face/success_feature.txt", real);
	HashCoding(real, hashcode);									//将128位的float数组哈希编码成16个Byte

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&g_hashcode, TOTAL_DATA * sizeof(HashCode));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMemcpyAsync(g_hashcode, hashcode, TOTAL_DATA * sizeof(HashCode), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	//cudaDeviceSynchronize waits for the kernel to finish, and returns
	//any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
	}
}

//读取选择的待测试的人脸特征向量，传入显存中
extern "C"
void read_face_test(int index)
{
	readTestFeature(ROOT_PATH + "face/test_face/success_feature.txt", test_r, index);
	HashCoding_test(test_r, test_hc);							//将128个float哈希编码成16个Byte
}

//筛选出相似度前k个的人脸
extern "C"
PathSimilarity * calc()
{
	IndexHamming * indexHamming = new IndexHamming[CHOOSEN_DATA];	//存储海明距离和对应索引的结构体数组
	IndexCosine * indexCosine = new IndexCosine[CHOOSEN_DATA];		//存储余弦距离和对应索引的结构体数组

	HashCode * g_test_hc;											//显存中存储待测人脸哈希码
	IndexHamming * g_indexHamming;									//显存中存储海明距离与索引

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&g_test_hc, 1 * sizeof(HashCode));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&g_indexHamming, TOTAL_DATA * sizeof(IndexHamming));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		///goto Error;
	}

	cudaStatus = cudaMemcpyAsync(g_test_hc, test_hc, 1 * sizeof(HashCode), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	dim3 threads(128);
	dim3 blocks(2);    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	//int GridSize = (TOTAL_DATA / HASH_PER_THREAD + blockSize - 1) / blockSize;


	//计算海明距离
	calc_HammingDis_Kernel << <blocks, threads >> >(g_hashcode, g_test_hc, g_indexHamming);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
	}

	cudaStatus = cudaMemcpy(indexHamming, g_indexHamming, CHOOSEN_DATA * sizeof(IndexHamming), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in hammingDis!");
		//goto Error;
	}

	//调用双调排序算法排序海明距离
	//将TOTAL_DATA个海明距离中每CHOSEN_DATA个排成有序
	for (int i = 2; i <= CHOOSEN_DATA; i <<= 1)
		//for (int i = 2; i <= TOTAL_DATA; i <<= 1)
	{
		//printf("Merge i=%d\n", i);
		bitonic_merge_Kernel << <blocks, threads >> >(g_indexHamming, i);
		for (int j = (i >> 1); j >= 2; j = j >> 1)
		{
			//printf("i=%d, j=%d\n", i, j);
			bitonic_half_cleaner_Kernel << <blocks, threads >> >(g_indexHamming, j);
		}
	}

	for (int i = 2; i <= TOTAL_DATA / CHOOSEN_DATA; i <<= 1)
	{
		head_tail_sort << <blocks, threads >> >(g_indexHamming, i);
		//head_tail_sort_single << <GridSize, blockSize >> >(g_indexHamming, i);
		//将CHOOSEN_DATA个数据排成有序
		head_choosen_sort << <blocks, threads >> >(g_indexHamming, i);
	}


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}

	//cudaDeviceSynchronize waits for the kernel to finish, and returns
	//any errors encountered during the launch.
	/**/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
	}

	cudaStatus = cudaMemcpy(indexHamming, g_indexHamming, CHOOSEN_DATA * sizeof(IndexHamming), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in hammingDis!");
		//goto Error;
	}

	
	//Error:
	cudaFree(g_indexHamming);
	cudaFree(g_test_hc);
	

	/*****************************************余弦距离筛选******************************/
	/**/
	//从CHOOSEN出来的结果中选出最后的进行精确检索
	//在CHOOSEN_DATA中做余弦距离运算
	float up = 0, down1 = 0, down2 = 0;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
		indexCosine[i].index = indexHamming[i].index;
		for (int j = 0; j < FEAT; j++)
		{
			up += test_r[j] * real[indexCosine[i].index*FEAT + j];
			down1 += test_r[j] * test_r[j];
			down2 += real[indexCosine[i].index*FEAT + j] * real[indexCosine[i].index*FEAT + j];

		}
		indexCosine[i].cosineDis = up / (sqrt(down1*down2));
	}

	//将余弦距离排序
	IndexCosine tem;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
		for (int j = i + 1; j < CHOOSEN_DATA; j++)
		{
			if (indexCosine[i].cosineDis < indexCosine[j].cosineDis)
			{
				tem = indexCosine[i];
				indexCosine[i] = indexCosine[j];
				indexCosine[j] = tem;
			}
		}
	}

	//将Top4中余弦距离相同的项，按索引号升序排序
 	for (int i = 0; i < face_num; i++)
	{
		for (int j = i + 1; j < face_num; j++)
		{
			//if (indexCosine[i].cosineDis == indexCosine[j].cosineDis)
			{
				if (indexCosine[i].index > indexCosine[j].index)
				{
					tem = indexCosine[i];
					indexCosine[i] = indexCosine[j];
					indexCosine[j] = tem;
				}
			}
		}
	}


	//读取人脸图片路径的txt，将索引号对应成文件路径
	//string * face_path = new string[face_num];
	PathSimilarity * pathSimilarity = new PathSimilarity[face_num];

	ifstream input_face(ROOT_PATH + "face/lib_face/success_image.txt");
	if (!input_face.is_open())
	{
		cout << "file open failure" << endl;
	}
	else
	{
		int count = 0;
		int index_face_num = 0;
		string tem;
		while (!input_face.eof() && count <= indexCosine[face_num - 1].index)
		{
			input_face >> tem;
			if (count == indexCosine[index_face_num].index)
			{
				pathSimilarity[index_face_num].path = "G:/MyQt/face/lib_face" + tem.erase(0, 38);
				pathSimilarity[index_face_num].similarity = indexCosine[index_face_num].cosineDis;
				index_face_num++;
			}
			count++;
		}
	}
	input_face.close();

	delete[] indexHamming;
	delete[] indexCosine;
	
	//system("pause");
	return pathSimilarity;
}

//回收显存资源
extern "C"
void revoke()
{
	cudaError_t cudaStatus;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}

	delete[] real;
	delete[] hashcode;
	delete[] test_r;
	delete[] test_hc;
	cudaFree(g_hashcode);
}


// Helper function for using CUDA to choose top-K in parallel.
cudaError_t calWithCuda(HashCode * hashcode, HashCode * test_hc, IndexHamming * indexHamming)
{
	HashCode * g_hashcode;								//GPU显存中存储1000000 个哈希码结构体的数组
	HashCode * g_test_hc;								//显存中1 个测试哈希码结构体
	IndexHamming * g_indexHamming;						//现存中存储海明距离与索引

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	//用于性能测试的事件
	/**/
	cudaEvent_t g_start, g_stop;
	cudaEventCreate(&g_start);
	cudaEventCreate(&g_stop);
	cudaEventRecord(g_start, 0);


	cudaStatus = cudaMalloc((void**)&g_hashcode, TOTAL_DATA * sizeof(HashCode));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&g_test_hc, 1 * sizeof(HashCode));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaStatus = cudaMalloc((void**)&g_indexHamming, TOTAL_DATA * sizeof(IndexHamming));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		///goto Error;
	}

	/**/
	cudaEventRecord(g_stop, 0);
	cudaEventSynchronize(g_stop);
	float malloc_timeused;
	cudaEventElapsedTime(&malloc_timeused, g_start, g_stop);
	printf("CUDA malloc time: %lf\n", malloc_timeused);


	cudaEventRecord(g_start, 0);
	// Copy input vectors from host memory to GPU buffers.

	cudaStatus = cudaMemcpyAsync(g_hashcode, hashcode, TOTAL_DATA * sizeof(HashCode), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	/*
	cudaStatus = cudaMemcpyAsync(g_hashcode, hashcode, TOTAL_DATA*FEAT / 8 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	//goto Error;
	}

	cudaStatus = cudaMemcpyAsync(g_hashcode + TOTAL_DATA*FEAT / 8 / 2, hashcode + TOTAL_DATA*FEAT / 8 / 2, TOTAL_DATA*FEAT / 8 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	//goto Error;
	}*/

	cudaStatus = cudaMemcpyAsync(g_test_hc, test_hc, 1 * sizeof(HashCode), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	/**/
	cudaEventRecord(g_stop, 0);
	cudaEventSynchronize(g_stop);
	float trans_timeused;
	cudaEventElapsedTime(&trans_timeused, g_start, g_stop);
	printf("data transport time: %lf\n", trans_timeused);


	// Launch a kernel on the GPU with one thread for each element.
	//int blockSize = 1024;      // The launch configurator returned block size
	dim3 threads(128);
	dim3 blocks(2);    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	//int GridSize = (TOTAL_DATA / HASH_PER_THREAD + blockSize - 1) / blockSize;

	//开始计时
	cudaEventRecord(g_start, 0);

	//calKernel<1024> << <GridSize, blockSize, sizeof(int)*blockSize >> >(g_hashcode, g_test_hc, g_hammingDis, g_index, g_id);

	//计算海明距离
	calc_HammingDis_Kernel << <blocks, threads >> >(g_hashcode, g_test_hc, g_indexHamming);


	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
	}

	cudaStatus = cudaMemcpy(indexHamming, g_indexHamming, CHOOSEN_DATA * sizeof(IndexHamming), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in hammingDis!");
		//goto Error;
	}


	//计算海明距离的TopK
	//topK_hanmming<< <blocks, threads >> >(g_indexHamming);

	/**/
	//调用双调排序算法排序海明距离
	//将TOTAL_DATA个海明距离中每CHOSEN_DATA个排成有序
	for (int i = 2; i <= CHOOSEN_DATA; i <<= 1)
		//for (int i = 2; i <= TOTAL_DATA; i <<= 1)
	{
		//printf("Merge i=%d\n", i);
		bitonic_merge_Kernel << <blocks, threads >> >(g_indexHamming, i);
		for (int j = (i >> 1); j >= 2; j = j >> 1)
		{
			//printf("i=%d, j=%d\n", i, j);
			bitonic_half_cleaner_Kernel << <blocks, threads >> >(g_indexHamming, j);
		}
	}

	/**/
	for (int i = 2; i <= TOTAL_DATA / CHOOSEN_DATA; i <<= 1)
	{
		head_tail_sort << <blocks, threads >> >(g_indexHamming, i);
		//head_tail_sort_single << <GridSize, blockSize >> >(g_indexHamming, i);
		//将CHOOSEN_DATA个数据排成有序
		head_choosen_sort << <blocks, threads >> >(g_indexHamming, i);
	}

	//调用归并排序排序海明距离
	/*
	for (int i = 2; i < TOTAL_DATA; i <<= 1)
	{

	}*/

	//计时结束并输出
	/**/
	cudaEventRecord(g_stop, 0);
	cudaEventSynchronize(g_stop);
	float g_timeused;
	cudaEventElapsedTime(&g_timeused, g_start, g_stop);
	printf("kernel calculate time: %lf\n", g_timeused);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}

	//cudaDeviceSynchronize waits for the kernel to finish, and returns
	//any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
	}


	cudaStatus = cudaMemcpy(indexHamming, g_indexHamming, CHOOSEN_DATA * sizeof(IndexHamming), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in hammingDis!");
		//goto Error;
	}

	//测试输出
	/*
	cout << "Top " << CHOOSEN_DATA << ":" << endl;

	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
	cout << indexHamming[i].index << " " << indexHamming[i].hammingDis << endl;
	}
	*/
	//Error:
	cudaFree(g_hashcode);
	cudaFree(g_test_hc);
	cudaFree(g_indexHamming);

	return cudaStatus;
}

extern "C"
void Topk(int index)
{

	HashCode * hashcode = new HashCode[TOTAL_DATA];				//TOTAL_DATA 个HashCode结构体 的哈希码
	Real * real = new float[TOTAL_DATA*FEAT];					//TOTAL_DATA*128个float 的实值
	HashCode * test_hc = new HashCode;							//待测试的哈希码
	Real * test_r = new float[FEAT];							//待测试的实值

	IndexHamming * indexHamming = new IndexHamming[CHOOSEN_DATA];//存储海明距离和对应索引的结构体数组
	IndexCosine * indexCosine = new IndexCosine[CHOOSEN_DATA];	//存储余弦距离和对应索引的结构体数组

	//随机产生128位float数组，编成哈希码
	//ProduceRandom(real);										//随机产生128位float数组，存储在real中
	//从文件中读取实值

	readFeature("G:/at_HUST/daily_HUST/17-4/14/lib_face/success_feature.txt", real);
	HashCoding(real, hashcode);									//将128位的float数组哈希编码成16个Byte

	//ProduceRandom_test(test_r);								//生成128个float，存储在test_r中
	readTestFeature("G:/at_HUST/daily_HUST/17-4/14//test_face/success_feature.txt", test_r, index);
	HashCoding_test(test_r, test_hc);							//将128个float哈希编码成16个Byte

	cudaError_t cudaStatus = calWithCuda(hashcode, test_hc, indexHamming);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return;
	}

	//从CHOOSEN出来的结果中选出最后的进行精确检索
	//在CHOOSEN_DATA中做余弦距离运算
	float up = 0, down1 = 0, down2 = 0;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
		indexCosine[i].index = indexHamming[i].index;
		//indexCosine[i].cosineDis = 0;
		for (int j = 0; j < FEAT; j++)
		{
			up += test_r[j] * real[indexCosine[i].index*FEAT + j];
			down1 += test_r[j] * test_r[j];
			down2 += real[indexCosine[i].index*FEAT + j] * real[indexCosine[i].index*FEAT + j];

		}
		indexCosine[i].cosineDis = up / (sqrt(down1*down2));
	}

	//Top-32的余弦距离输出
	/*
	cout << "Top " << CHOOSEN_DATA << "(余弦排序前):" << endl;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
	cout << indexCosine[i].index << " " << indexCosine[i].cosineDis << endl;
	}*/

	//将余弦距离排序
	IndexCosine tem;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
		for (int j = i + 1; j < CHOOSEN_DATA; j++)
		{
			if (indexCosine[i].cosineDis < indexCosine[j].cosineDis)
			{
				tem = indexCosine[i];
				indexCosine[i] = indexCosine[j];
				indexCosine[j] = tem;
			}
		}
	}

	//将Top4中余弦距离相同的项，按索引号升序排序
	for (int i = 0; i < face_num; i++)
	{
		for (int j = i + 1; j < face_num; j++)
		{
			//if (indexCosine[i].cosineDis == indexCosine[j].cosineDis)
			{
				if (indexCosine[i].index > indexCosine[j].index)
				{
					tem = indexCosine[i];
					indexCosine[i] = indexCosine[j];
					indexCosine[j] = tem;
				}
			}
		}
	}

	//Top-32的余弦距离输出
	cout << "Top " << CHOOSEN_DATA << "(余弦排序后):" << endl;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
		cout << indexCosine[i].index << " " << indexCosine[i].cosineDis << endl;
	}

	//读取人脸图片路径的txt，将索引号对应成文件路径
	string * face_path = new string[face_num];

	ifstream input_face("G:/MyQt/face/lib_face/success_image.txt");
	if (!input_face.is_open())
	{
		cout << "file open failure" << endl;
	}
	else
	{
		int count = 0;
		int index_face_num = 0;
		string tem;
		while (!input_face.eof() && count <= indexCosine[face_num - 1].index)
		{
			input_face >> tem;
			if (count == indexCosine[index_face_num].index)
			{
				face_path[index_face_num] = "G:/MyQt/face/lib_face" + tem.erase(0, 38);
				index_face_num++;
			}
			count++;
		}
	}
	input_face.close();

	
	//将人脸路径和余弦距离写入txt文档中
	ofstream output("G:/MyQt/Top4.txt");
	for (int i = 0; i < face_num; i++)
	{
		output << face_path[i] << " " << indexCosine[i].cosineDis << endl;
	}
	output.close();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}
	

	//delete[] param;
	delete test_hc;
	delete[] test_r;
	delete[] hashcode;
	delete[] real;
	delete[] indexHamming;
	delete[] indexCosine;

	//system("pause");
	return;
}




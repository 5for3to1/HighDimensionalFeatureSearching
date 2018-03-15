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

//�洢����ѡ��������ƶ���ͼƬ·��
struct PathSimilarity
{
	string path;
	float similarity;
};

//�洢��������������Ľṹ��
struct IndexHamming
{
	int index;
	unsigned short hammingDis;
};

//�洢���Ҿ���������Ľṹ��
struct IndexCosine
{
	int index;
	float cosineDis;
};

//�洢��ϣ��Ľṹ��
struct HashCode
{
	unsigned char codeTtem[64];
};

#define TOTAL_DATA 256										//1000*1000
#define FEAT 512											//����������λ��

#define CHOOSEN_DATA 32										//��������ɸѡ��������
#define face_num 5											//����ѡ�������������



/**************************************************************/
/*****************���������ļ��е����·��********************/
string ROOT_PATH = "G:/MyQt/";								//����ͼƬ�͹������ڵ�·��


Real * real = new float[TOTAL_DATA*FEAT];					//�洢������ʵֵ
HashCode * hashcode = new HashCode[TOTAL_DATA];				//TOTAL_DATA ��HashCode�ṹ�� �Ĺ�ϣ��

Real * test_r = new float[FEAT];							//�����Ե�ʵֵ
HashCode * test_hc = new HashCode;

HashCode * g_hashcode;										//�Դ��д洢�������ϣ��


//����P��Q�ĺ�������
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

//len�����ݣ���i��������� len-i �����ݽ��бȽ�
__global__ void bitonic_merge_Kernel(IndexHamming * indexHamming, int len)
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;			//��ǰ�̺߳�
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / 2)
	{
		int offset = index % (len / 2);								/*������ƫ��*/
		int base = index / (len / 2);								/*���ڵ������*/

		int source = base*len + offset;								/*�̶߳�Ӧ�����ݵ��±�*/
		int target = base*len + len - offset - 1;

		IndexHamming temp;													//���ڽ������м����
		if (indexHamming[source].hammingDis >  indexHamming[target].hammingDis)
		{
			temp = indexHamming[source];
			indexHamming[source] = indexHamming[target];
			indexHamming[target] = temp;
		}
		__syncthreads();
	}
}

//len�����ݣ���i��������� len-i �����ݽ��бȽ�
__global__ void head_tail_sort(IndexHamming * indexHamming, int stride)
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;				//��ǰ�̺߳�
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / stride)
	{
		int base = index / CHOOSEN_DATA * stride;					/*���ڵ������*/
		int offset = index % CHOOSEN_DATA;							/*������ƫ��*/

		int source = base*CHOOSEN_DATA + offset;					/*�̶߳�Ӧ�����ݵ��±�*/
		int target = (base + stride / 2)*CHOOSEN_DATA + CHOOSEN_DATA - offset - 1;

		//IndexHamming temp;													//���ڽ������м����
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
	//int index = threadIdx.x + blockIdx.x * blockDim.x;				//��ǰ�̺߳�
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

//��CHOOSEN_DATA�������ų������
__global__ void head_choosen_sort(IndexHamming * indexHamming, int stride)		//len=CHOOSEN_DATA/2;
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;			//��ǰ�̺߳�
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / (stride * 2))
	{
		int base_const = index / (CHOOSEN_DATA / 2);					/*���ڵ������*/
		int offset_const = index % (CHOOSEN_DATA / 2);

		IndexHamming temp;										//���ڽ������м����

		for (int len = CHOOSEN_DATA >> 1; len >= 1; len >>= 1)
		{
			int base = offset_const / len;									/*���ڵ������*/
			int offset = offset_const % len;								/*������ƫ��*/

			int source = base * len * 2 + offset + CHOOSEN_DATA * stride * base_const;		/*�̶߳�Ӧ�����ݵ��±�*/
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

//len�����ݣ���i��������� i+len/2 �����ݽ��бȽ�
__global__ void bitonic_half_cleaner_Kernel(IndexHamming * indexHamming, int len)
{
	//int index = threadIdx.x + blockIdx.x * blockDim.x;			//��ǰ�̺߳�
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	if (index < TOTAL_DATA / 2)
	{
		int offset = index % (len / 2);								/*������ƫ��*/
		int base = index / (len / 2);								/*���ڵ������*/

		index = base*len + offset;									/*�̶߳�Ӧ�����ݵ��±�*/
		int target = index + len / 2;

		IndexHamming temp;											//���ڽ������м����
		if (indexHamming[index].hammingDis >  indexHamming[target].hammingDis)
		{
			temp = indexHamming[index];
			indexHamming[index] = indexHamming[target];
			indexHamming[target] = temp;
		}
		__syncthreads();
	}
}


/*******************************************��������*****************************************/

void ProduceRandom(Real * real)					//α������� TOTAL_DATA*FEAT ��float
{
	for (int i = 0; i < TOTAL_DATA*FEAT; i++)
	{
		real[i] = rand() % 20 - 10;				//����-10��10֮��������
	}
}

void ProduceRandom_test(Real * test_r)				//α�������һ��FEATλ��float����
{
	for (int i = 0; i < FEAT; i++)
	{
		test_r[i] = rand() % 20 - 10;					//����-10��10֮��������
	}
}


//ר���ڽ������128λfloat��ʵֵ ���� �ɰ����16��Byte�Ĺ�ϣ��
void HashCoding(Real * real, HashCode * hashcode)
{
	for (int i = 0; i < TOTAL_DATA; i++)
	{

		for (int j = 0; j <FEAT; j++)
		{
			if (j % 8 == 0)												//ÿ����8��float��һ����ϣ��BYTE��ʼ��
			{
				hashcode[i].codeTtem[j / 8] = 0;
			}
			if (real[j + i*FEAT] > 0)
			{
				hashcode[i].codeTtem[j / 8] += pow(2, 7 - (j % 8));		//��ǰ8��float�����һ��Byte
			}
		}
	}
}

void HashCoding_test(Real * test_r, HashCode * test_hc)
{
	for (int i = 0; i <FEAT; i++)
	{
		if (i % 8 == 0)												//ÿ����8��float��һ����ϣ���ʼ��һ��
		{
			test_hc->codeTtem[i / 8] = 0;
		}
		if (test_r[i]>0)
		{
			test_hc->codeTtem[i / 8] += pow(2, 7 - (i % 8));		//��ǰ8��float�����һ��Byte
		}
	}
}

//��ȡlib����������
void readFeature(string fileName, Real * real)
{

	ifstream in(fileName);				//�ļ������ļ�
	if (!in.is_open())
	{
		cout << "�ļ���ʧ�ܣ�" << endl;
	}
	else
	{
		int count = 0;
		while (!in.eof())
		{
			in >> real[count];
			count++;
		}
		//cout << "��������" << count/FEAT << endl;
	}
}

//��ȡtest����������
void readTestFeature(string fileName, Real * test_r, int index)
{
	ifstream in(fileName);				//�ļ������ļ�
	if (!in.is_open())
	{
		cout << "�ļ���ʧ�ܣ�" << endl;
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


//��ȡ����������������������Դ���
extern "C"
void red_face_lib()
{
	readFeature(ROOT_PATH + "face/lib_face/success_feature.txt", real);
	HashCoding(real, hashcode);									//��128λ��float�����ϣ�����16��Byte

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

//��ȡѡ��Ĵ����Ե��������������������Դ���
extern "C"
void read_face_test(int index)
{
	readTestFeature(ROOT_PATH + "face/test_face/success_feature.txt", test_r, index);
	HashCoding_test(test_r, test_hc);							//��128��float��ϣ�����16��Byte
}

//ɸѡ�����ƶ�ǰk��������
extern "C"
PathSimilarity * calc()
{
	IndexHamming * indexHamming = new IndexHamming[CHOOSEN_DATA];	//�洢��������Ͷ�Ӧ�����Ľṹ������
	IndexCosine * indexCosine = new IndexCosine[CHOOSEN_DATA];		//�洢���Ҿ���Ͷ�Ӧ�����Ľṹ������

	HashCode * g_test_hc;											//�Դ��д洢����������ϣ��
	IndexHamming * g_indexHamming;									//�Դ��д洢��������������

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


	//���㺣������
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

	//����˫�������㷨����������
	//��TOTAL_DATA������������ÿCHOSEN_DATA���ų�����
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
		//��CHOOSEN_DATA�������ų�����
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
	

	/*****************************************���Ҿ���ɸѡ******************************/
	/**/
	//��CHOOSEN�����Ľ����ѡ�����Ľ��о�ȷ����
	//��CHOOSEN_DATA�������Ҿ�������
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

	//�����Ҿ�������
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

	//��Top4�����Ҿ�����ͬ�������������������
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


	//��ȡ����ͼƬ·����txt���������Ŷ�Ӧ���ļ�·��
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

//�����Դ���Դ
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
	HashCode * g_hashcode;								//GPU�Դ��д洢1000000 ����ϣ��ṹ�������
	HashCode * g_test_hc;								//�Դ���1 �����Թ�ϣ��ṹ��
	IndexHamming * g_indexHamming;						//�ִ��д洢��������������

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	//�������ܲ��Ե��¼�
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

	//��ʼ��ʱ
	cudaEventRecord(g_start, 0);

	//calKernel<1024> << <GridSize, blockSize, sizeof(int)*blockSize >> >(g_hashcode, g_test_hc, g_hammingDis, g_index, g_id);

	//���㺣������
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


	//���㺣�������TopK
	//topK_hanmming<< <blocks, threads >> >(g_indexHamming);

	/**/
	//����˫�������㷨����������
	//��TOTAL_DATA������������ÿCHOSEN_DATA���ų�����
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
		//��CHOOSEN_DATA�������ų�����
		head_choosen_sort << <blocks, threads >> >(g_indexHamming, i);
	}

	//���ù鲢��������������
	/*
	for (int i = 2; i < TOTAL_DATA; i <<= 1)
	{

	}*/

	//��ʱ���������
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

	//�������
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

	HashCode * hashcode = new HashCode[TOTAL_DATA];				//TOTAL_DATA ��HashCode�ṹ�� �Ĺ�ϣ��
	Real * real = new float[TOTAL_DATA*FEAT];					//TOTAL_DATA*128��float ��ʵֵ
	HashCode * test_hc = new HashCode;							//�����ԵĹ�ϣ��
	Real * test_r = new float[FEAT];							//�����Ե�ʵֵ

	IndexHamming * indexHamming = new IndexHamming[CHOOSEN_DATA];//�洢��������Ͷ�Ӧ�����Ľṹ������
	IndexCosine * indexCosine = new IndexCosine[CHOOSEN_DATA];	//�洢���Ҿ���Ͷ�Ӧ�����Ľṹ������

	//�������128λfloat���飬��ɹ�ϣ��
	//ProduceRandom(real);										//�������128λfloat���飬�洢��real��
	//���ļ��ж�ȡʵֵ

	readFeature("G:/at_HUST/daily_HUST/17-4/14/lib_face/success_feature.txt", real);
	HashCoding(real, hashcode);									//��128λ��float�����ϣ�����16��Byte

	//ProduceRandom_test(test_r);								//����128��float���洢��test_r��
	readTestFeature("G:/at_HUST/daily_HUST/17-4/14//test_face/success_feature.txt", test_r, index);
	HashCoding_test(test_r, test_hc);							//��128��float��ϣ�����16��Byte

	cudaError_t cudaStatus = calWithCuda(hashcode, test_hc, indexHamming);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return;
	}

	//��CHOOSEN�����Ľ����ѡ�����Ľ��о�ȷ����
	//��CHOOSEN_DATA�������Ҿ�������
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

	//Top-32�����Ҿ������
	/*
	cout << "Top " << CHOOSEN_DATA << "(��������ǰ):" << endl;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
	cout << indexCosine[i].index << " " << indexCosine[i].cosineDis << endl;
	}*/

	//�����Ҿ�������
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

	//��Top4�����Ҿ�����ͬ�������������������
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

	//Top-32�����Ҿ������
	cout << "Top " << CHOOSEN_DATA << "(���������):" << endl;
	for (int i = 0; i < CHOOSEN_DATA; i++)
	{
		cout << indexCosine[i].index << " " << indexCosine[i].cosineDis << endl;
	}

	//��ȡ����ͼƬ·����txt���������Ŷ�Ӧ���ļ�·��
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

	
	//������·�������Ҿ���д��txt�ĵ���
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




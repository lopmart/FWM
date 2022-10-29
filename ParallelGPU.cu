
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <string>


const float pi = 3.141592653589793;
const int sizeD = 144; 
const int N = 240;

using namespace std;


__global__ void kernelFWM(float* Pfwm, float* eta, float* nd_k, float pt0, float nn, float X, float leff, float mm, float aeff, float a, float La, float lc, float c, float df, float D,
	float dD_dL, float dijk) {
	float pi = 3.141592653589793;
	int fwm = (blockIdx.x * blockDim.x) + threadIdx.x;
	float PFijk = 0; float nd = 0;
	PFijk = 0;
	nd = 0;
	float fo = c / lc;
	float dbijk = 0, nijk = 0;
	float kaux = 0;
	int s = 0;
	for (int k = 0; k < N; k++) {
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				s = i + j - k;
				float I = i + 1; 
				float J = j + 1; 
				float K = k + 1;  
				if ((s == fwm) && (j != k) && (i != k) && (I >= J)) {
					dbijk = (((2 * pi * pow(lc, 2)) / c) * abs(I - K) * abs(J - K) * (pow(df, 2)) * (D + df * (dD_dL * pow(lc, 2) / (2 * c)) * (I - K + J - K)));
					nijk = ((pow(a, 2) / (pow(a, 2) + pow(dbijk, 2))) * (1 + (4 * exp(-a * La)) * pow((sin(dbijk * (La / 2))), 2) / pow((1 - (exp(-a * La))), 2)));
					kaux = (32 * X * (pow(pi, 3)) / (c * lc * (pow(nn, 2))));
					dijk = 6;
					PFijk = PFijk + (pow(kaux, 2) * pow(pt0, 3) * (exp(-a * La)) * (mm * (pow((leff / aeff), 2))) * (nijk * (pow(dijk, 2))));
					nd = nd + nijk * pow(dijk, 2);
				}
			}
		}
	}

	Pfwm[fwm] = 10 * log10(PFijk / 1e-3);
	eta[fwm] = dbijk;
	nd_k[fwm] = nd;

	///---------------------------------------------------------------------------
}
//Kernel master
__global__ void kernelD(float* Lmax, float* Ld, float* Po, float* Dv, float* cl, float kaux, float pt0, float nn, float X, float leff, float mm, float aeff, float a, float La, float lc, float c, float df,
	float dD_dL, float dijk) {

	int DD = (blockIdx.x * blockDim.x) + threadIdx.x;
	float D = Dv[DD];
	float* Pfwm;
	Pfwm = new float[N];
	float* eta;
	eta = new float[N];
	float* nd_k;
	nd_k = new float[N];
	for (int in = 0; in < N; in++) {
		Pfwm[in] = 0;
		eta[in] = 0;
		nd_k[in] = 0;
	}
	kernelFWM <<<1, N >>>(Pfwm, eta, nd_k, pt0, nn, X, leff, mm, aeff, a, La, lc, c, df, D, dD_dL, dijk);
	
	cudaDeviceSynchronize();
	float nd_kmax = nd_k[0];
	int pos_kmax = 0;
	float ym_max = Pfwm[0];
	int pos_ym = 0;
	float testSum = 0; 

	for (int i = 0; i < N; i++)
	{
		if (nd_k[i] > nd_kmax)
		{ nd_kmax = nd_k[i]; pos_kmax = i;
		}
	if (Pfwm[i] > ym_max)
		{
			ym_max = Pfwm[i];
			pos_ym = i;
		}
		testSum = testSum + Pfwm[i];
	}
	float B = 2.5e9;
	float Bo = 4 * B;
	float SNRo = 100;
	float SNRfwm = SNRo;
	float nsp = 1;
	float E = 1;
	float h = 6.63e-34;
	
	float L_max1 = (aeff * pow(La, 2)) / (2 * SNRo * h * (c / cl[pos_ym]) * nsp * Bo * (E * exp(a * La) - 1) * leff);
	float  L_max2 = sqrt(1 / (SNRfwm * pow(kaux, 2) * nd_kmax));
	Lmax[DD] = sqrt(L_max1 * L_max2) * 0.001;
	Ld[DD] = c / (2 * pow(B, 2) * pow(cl[pos_ym], 2) * abs(Dv[DD]));
    Po[DD] = 2 * SNRo * h * (c / cl[pos_ym]) * nsp * Bo * (E * exp(a * La) - 1) * (1 / La) * Lmax[DD];
    delete[] Pfwm;
	delete[] eta;
	delete[] nd_k;

}

int main() {
	ofstream times;

	times.open("timewith12Dfijo.txt");

	cout << "iniciando" << endl;

	float sumTiempo = 0;
	float df, dl = 0, ch = 0, pt0 = 0, L = 0, X = 0;

	float dD_dL = 0, leff = 0, aeff = 0;
	float c = 0, lc = 0;
	float a = 0;
	float delta = 0, amplif_sep = 0, L_system = 0, mm = 0, dijk = 6, La = 0;
	float at0 = 0.25, nn = 1.45;
	float kaux = 0;

	float* cl;
	cl = new float[N];

	float cf[N];
	float pt[N];
	float at[N];


	for (int in = 0; in < N; in++) {
		cl[in] = 0;
		cf[in] = 0;
		pt[in] = 0;
		at[in] = 0;
	}

	
	float* Dv;
	Dv = new float[sizeD];

	float* Lmax;
	Lmax = new float[sizeD];

	float* Ld;
	Ld = new float[sizeD];

	float* Po;
	Po = new float[sizeD];

	//----------------------

	for (int i = 0; i < sizeD; i++)
	{
		Dv[i] = i * 0.0625;

		Lmax[i] = 0;
		Ld[i] = 0;
		Po[i] = 0;
	}

	lc = 1550E-9;
	c = 3E8;
	df = 3.75e12 / (N - 1);
	dl = lc - c * lc / (c + df * lc);
	ch = lc - (1 + floor(N / 2)) * dl;
	pt0 = 1e-3;
	amplif_sep = 75;
	L_system = 75;
	mm = L_system / amplif_sep;
	L = L_system * 1e3;
	X = 6e-14;
	dD_dL = 0.09e3;
	La = amplif_sep * 1e3;
	a = (at0 / 4.343) * 1e-3;
	leff = (1 - exp(-a * La)) * (1 / a);
	aeff = 53e-12;
	float aa = 0;
	for (int i = 0; i < N; i++) {
		delta = delta + ch + dl;
		cl[i] = delta;
		ch = 0;
		at[i] = at0;
		pt[i] = pt0;

	}
	cf[0] = 3e8 / cl[0];

	for (int i = 1; i < N; i++) {
		cf[i] = cf[i] + (i)*df;
	}

	kaux = (32 * X * (pow(pi, 3)) / (c * lc * (pow(nn, 2))));
    cudaEvent_t start, stop; //medicion de tiempo
	float time;
	size_t bytes;
	size_t bytes2;
	bytes = sizeof(float) * sizeD;
	bytes2 = sizeof(float) * N;
	float* gpu_Lmax;
	float* gpu_Ld;
	float* gpu_Po;
	float* gpu_Dv;
	float* gpu_cl;
	cudaMalloc((void**)&gpu_Lmax, bytes);
	cudaMalloc((void**)&gpu_Ld, bytes);
	cudaMalloc((void**)&gpu_Po, bytes);
	cudaMalloc((void**)&gpu_Dv, bytes);
	cudaMalloc((void**)&gpu_cl, bytes2);

	cudaMemcpy(gpu_Lmax, Lmax, bytes, cudaMemcpyHostToDevice); 
	cudaMemcpy(gpu_Ld, Ld, bytes, cudaMemcpyHostToDevice); 
	cudaMemcpy(gpu_Po, Po, bytes, cudaMemcpyHostToDevice); 
	cudaMemcpy(gpu_Dv, Dv, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_cl, cl, bytes2, cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	kernelD <<<sizeD, 1 >>>(gpu_Lmax, gpu_Ld, gpu_Po, gpu_Dv, gpu_cl, kaux, pt0, nn, X, leff, mm, aeff, a, La, lc, c, df, dD_dL, dijk);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	sumTiempo = sumTiempo + time;  

	cudaMemcpy(Lmax, gpu_Lmax, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(Ld, gpu_Ld, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(Po, gpu_Po, bytes, cudaMemcpyDeviceToHost);

	cudaFree(gpu_Lmax);
	cudaFree(gpu_Ld);
	cudaFree(gpu_Po);
    cout << "\n\nN= " << N << "  " << "D=" << sizeD << " duration= " << sumTiempo << " ms" << endl;

	times << "{" << sumTiempo << "}," << endl;

	cout << "clock resolution " << sumTiempo / 1000 << " seg" << endl;
	

	ofstream arcLmax, arcLd, arcDD, arcPo;

	char nameLmax[20] = "Lmax";
	strcat_s(nameLmax, ".txt");

	char nameLd[20] = "Ld";
	strcat_s(nameLd, ".txt");

	char nameDD[20] = "DD";
	strcat_s(nameDD, ".txt");

	char namePo[20] = "Po";
	strcat_s(namePo, ".txt");

	arcLmax.open(nameLmax);

	arcLd.open(nameLd);

	arcDD.open(nameDD);

	arcPo.open(namePo);

	if (arcLmax.fail()) { printf("file Lmax fail opening"); exit(1); }
	if (arcLd.fail()) { printf("file Ld fail opening"); exit(1); }
	if (arcDD.fail()) { printf("file DD fail opening"); exit(1); }
	if (arcPo.fail()) { printf("file Po fail opening"); exit(1); }

	for (int i = 0; i < sizeD; i++) {

		arcDD << fixed << setprecision(10) << Dv[i] / 1e-6;
		arcDD << "\n";

		arcLmax << fixed << setprecision(10) << Lmax[i];
		arcLmax << "\n";

		arcLd << fixed << setprecision(10) << Ld[i] * 0.001;
		arcLd << "\n";

		arcPo << fixed << setprecision(10) << Po[i] * 1e3;
		arcPo << "\n";

	}

	arcDD.close();
	arcLmax.close();
	arcLd.close();
	arcPo.close();

	delete[] Lmax;
	delete[] Ld;
	delete[] Po;
	delete[] Dv;
	delete[] cl;
	times.close();
	return 0;
}
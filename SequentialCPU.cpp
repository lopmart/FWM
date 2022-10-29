
#include <iostream>
#include <ctime>
#include <fstream>
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>



double pi = 3.141592653589793;
const int sizeD = 144;
const int N = 240; 

using namespace std;
int main(int argc, const char* argv[]) {
	

	ofstream times;

	times.open("timewith12Dfijo.txt");
   
		
		auto start = chrono::high_resolution_clock::now();
		double df, dl = 0, ch = 0, pt0 = 0, L = 0, X = 0, kaux = 0;
		double dD_dL = 0, leff = 0, aeff = 0;
		double c = 0, lc = 0, fo = 0;
		double a = 0;
		double delta = 0, amplif_sep = 0, L_system = 0, mm = 0, dijk = 6, La = 0;
		double at0 = 0.25, nn = 1.45;
		double cl[N];
		double cf[N];
		double pt[N];
		double at[N];
		for (int in = 0; in < N; in++) {
			cl[in] = 0;
			cf[in] = 0;
			pt[in] = 0;
			at[in] = 0;
		}

		double Dv[sizeD];
		double Lmax[sizeD];
		double Ld[sizeD];
		double Po[sizeD];

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
		double aa = 0;
		for (int i = 0; i < N; i++) {
			delta = delta + ch + dl;
			cl[i] = delta;
			ch = 0;
			at[i] = at0;
			pt[i] = pt0;

		}
		cf[0] = 3e8 / cl[0];

		for (int i = 1; i < N; i++) {
			cf[i] = cf[i] + (i)* df;
		}
		for (int DD = 0; DD < sizeD; DD++)
		{
			double Pfwm[N];
			double eta[N];
			double nd_k[N];
			for (int in = 0; in < N; in++) {
				Pfwm[in] = 0;
				eta[in] = 0;
				nd_k[in] = 0;
			}
		double D = Dv[DD];
			for (int fwm = 0; fwm < N; fwm++)
			{
				double PFijk = 0; double nd = 0;
				PFijk = 0;
				nd = 0;
				fo = c / lc;
				double dbijk = 0, nijk = 0;
				kaux = 0;
				{
					int s = 0;
					for (int k = 0; k < N; k++) {
						for (int j = 0; j < N; j++) {
							for (int i = 0; i < N; i++) {
								s = i + j - k;
								if ((s == fwm) && (j != k) && (i != k)) {
									double I = (double)i + 1;  //0
									double J = (double)j + 1;  //1
									double K = (double)k + 1;  //2
									if (I >= J) {
										dbijk = (((2 * pi * lc*lc) / c) * abs(I - K) * abs(J - K) * (df*df) * (D + df * (dD_dL * (lc*lc)/ (2 * c)) * (I - K + J - K)));
										nijk = ((pow(a, 2) / (pow(a, 2) + pow(dbijk, 2))) * (1 + (4 * exp(-a * La)) * pow((sin(dbijk * (La / 2))), 2) / pow((1 - (exp(-a * La))), 2)));

										kaux = (32 * X * (pow(pi, 3)) / (c * lc * (pow(nn, 2))));
										dijk = 6;
										PFijk = PFijk + (pow(kaux, 2) * pow(pt0, 3) * (exp(-a * La)) * (mm * (pow((leff / aeff), 2))) * (nijk * (pow(dijk, 2))));
										nd = nd + nijk * pow(dijk, 2);
									}
								}
							}
						}
					}

					Pfwm[fwm] = 10 * log10(PFijk / 1e-3);
					eta[fwm] = dbijk;
					nd_k[fwm] = nd;
				}
			}
			double nd_kmax = nd_k[0];
			int pos_kmax = 0;
			double ym_max = Pfwm[0];
			int pos_ym = 0;
			double testSum = 0;  
			for (int i = 0; i < N; i++)
			{
				if (nd_k[i] > nd_kmax)
				{
					nd_kmax = nd_k[i];
					pos_kmax = i;
				}

				if (Pfwm[i] > ym_max)
				{
					ym_max = Pfwm[i];
					pos_ym = i;
				}
				testSum = testSum + Pfwm[i];
			}
			double B = 2.5e9;
			double Bo = 4 * B;
			double SNRo = 100;
			double SNRfwm = SNRo;
			double nsp = 1;
			double E = 1;
			double h = 6.63e-34;

			double L_max1 = (aeff * pow(La, 2)) / (2 * SNRo * h * (c / cl[pos_ym]) * nsp * Bo * (E * exp(a * La) - 1) * leff);
			double  L_max2 = sqrt(1 / (SNRfwm * pow(kaux, 2) * nd_kmax));
			Lmax[DD] = sqrt(L_max1 * L_max2) * 0.001;
			Ld[DD] = c / (2 * pow(B, 2) * pow(cl[pos_ym], 2) * abs(Dv[DD]));
			Po[DD] = 2 * SNRo * h * (c / cl[pos_ym]) * nsp * Bo * (E * exp(a * La) - 1) * (1 / La) * Lmax[DD];

		} 
		 

		auto end = chrono::high_resolution_clock::now();
		auto elapsed = end - start;
		cout << " duration= " << chrono::duration_cast<chrono::milliseconds>(elapsed).count() << " ms" << endl;

		
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

	
	times.close();
	
	return 0;
}

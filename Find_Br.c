#include <stdio.h>
#include "math.h"
#include <complex.h>
#include <fftw3.h>
#include "Find_Br.h"
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>


double Random(double min, double max)
{
    return (double)(rand())/RAND_MAX*(max - min) + min;
}



void Set_data()
{
	Cx = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	Ck = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	CCx = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	CCk = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	Cdx= (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));

	Cdk = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	CkNew = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	NL = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	M = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	CCCdx = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));

	K = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	CCCdk = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	KCCx = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	KCCk = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));    
	KCCCx = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));

	KCCCk = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	IK = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));
	Ckini = (fftw_complex*) fftw_malloc(N * sizeof(fftw_complex));

	Iter = 0;

	Dk = 2.0 * M_PI / (b - a);
	k0 = 100;
	V = 0.5 * sqrt(g/(k0 * Dk));
	w0 = sqrt (g * k0 * Dk);

	delta = 0.002;
	W = w0 - V * k0 * Dk + delta;
	A = sqrt(delta/(k0 * k0 * Dk * Dk));

	for ( int j = 0; j < N; j++)
	{
		Ck[j] = 0.0 + I * 0.0;
	} 

	Ck[k0 - 1] = 0.01 * A + I * 0.0;
	Ck[k0 + 1] = 0.01 * A + I * 0.0;
	Ck[k0] = A + I * 0.0;

	//  ik massive initializing

	for ( int j = N/2 + 1; j < N; j++)
	{
		IK[j] = 0.0 + I * 0.0;
		K[j] = -(j - N) * Dk;
		M[j] = 0.0 + I * 0.0;
	} 

	for ( int j = 0; j < N/2; j++)
	{
		IK[j] = I * j * Dk;
		K[j] = j * Dk;
		M[j] = W + V * j * Dk - sqrt(g * j * Dk);
	}

	IK[N/2] = 0.0 + I * 0.0;	
	K[N/2] = 0.0 + I * 0.0;
	M[N/2] = 0.0 + I * 0.0;

	printf("Initial data were successfully set \n\n");
}


void Closing()
{
	fftw_free(Cx);
	fftw_free(Ck);
	fftw_free(CCx);
	fftw_free(CCk);
	fftw_free(Cdx);

	fftw_free(Cdk);
	fftw_free(CkNew);
	fftw_free(NL);
	fftw_free(M);

	fftw_free(K);
	fftw_free(CCCdk);
	fftw_free(KCCx);
	fftw_free(KCCk);
	fftw_free(KCCCx);

	fftw_free(KCCCk);
	fftw_free(CCCdx);
	fftw_free(IK);
	fftw_free(Ckini);

}


void CreatePlans()
{
	Cx_2_Ck = fftw_plan_dft_1d(N,&Cx[0],&Ck[0],FFTW_FORWARD,FFTW_ESTIMATE);
 	Ck_2_Cx_ = fftw_plan_dft_1d(N,&Ck[0],&Cx[0],FFTW_BACKWARD,FFTW_ESTIMATE);
 	CCx_2_CCk = fftw_plan_dft_1d(N,&CCx[0],&CCk[0],FFTW_FORWARD,FFTW_ESTIMATE);

 	Cdk_2_Cdx_ = fftw_plan_dft_1d(N,&Cdk[0],&Cdx[0],FFTW_BACKWARD,FFTW_ESTIMATE);
 	KCCk_2_KCCx_ = fftw_plan_dft_1d(N,&KCCk[0],&KCCx[0],FFTW_BACKWARD,FFTW_ESTIMATE);
 	KCCCx_2_KCCCk = fftw_plan_dft_1d(N,&KCCCx[0],&KCCCk[0],FFTW_FORWARD,FFTW_ESTIMATE);
 	CCCdx_2_CCCdk = fftw_plan_dft_1d(N,&CCCdx[0],&CCCdk[0],FFTW_FORWARD,FFTW_ESTIMATE);
}

void DestroyPlans()
{
	fftw_destroy_plan(Cx_2_Ck);
	fftw_destroy_plan(Ck_2_Cx_);
	fftw_destroy_plan(CCx_2_CCk);
	fftw_destroy_plan(Cdk_2_Cdx_);
	fftw_destroy_plan(CCCdx_2_CCCdk);
	fftw_destroy_plan(KCCCx_2_KCCCk);
	fftw_destroy_plan(KCCk_2_KCCx_);

}

void Normalize(fftw_complex *Arr_x)
{
	for (int j = 0; j < N; j++)
	{
		Arr_x[j] = Arr_x[j]/N;
	}

	Arr_x[N/2] = 0.0 + I * 0.0;
}

void AbsSqrCC()
{
	for (int j = 0; j < N; j++)
	{
		CCx[j] = Cx[j] * conj(Cx[j]);
	}
}

void NonLin1()
{
	for (int j = 0; j < N; j++)
	{
		Cdk[j] = IK[j] * Ck[j];
	}

	fftw_execute(Cdk_2_Cdx_);

	for (int j = 0; j < N; j++)
	{
			CCCdx[j] = Cx[j] * conj(Cx[j]) * Cdx[j];
	}

	fftw_execute(CCCdx_2_CCCdk);
	Normalize(&CCCdk[0]);
}

void NonLin2()
{
	fftw_execute(CCx_2_CCk);

	Normalize(&CCk[0]);

	for (int j = 0; j < N; j++)
	{
		KCCk[j] = K[j] * CCk[j];
	}

	fftw_execute(KCCk_2_KCCx_);

	for (int j = 0; j < N; j++)
	{
		KCCCx[j] = KCCx[j] * Cx[j];
	}

	fftw_execute(KCCCx_2_KCCCk);

	Normalize(&KCCCk[0]);


}

double Sc1()
{
	double Summ = 0.0;

	for ( int j = 0; j < N/2; j++)
	{
		Summ = Summ + creal(Ck[j]) * creal(NL[j]);
	}

	return Summ;
}

double Sc2()
{
	double Summ = 0.0;

	for ( int j = 0; j < N/2; j++)
	{
		Summ = Summ + creal(Ck[j]) * creal(M[j]) * creal(Ck[j]);
	}
	return Summ;
}

void Breather()
{
	for ( int j = 0; j < N; j++)
	{
		Ckini[j] = Ck[j];
	}

	fftw_execute(Ck_2_Cx_);

	AbsSqrCC();



	NonLin1();

	NonLin2();

	for ( int j = 0; j < N/2; j++)
	{
		NL[j] = -IK[j] * CCCdk[j] + I * IK[j] * KCCCk[j];
	}

	for ( int j = N/2; j < N; j++)
	{
		NL[j] = 0.0 + I * 0.0;
	}

	for ( int j = 0; j < N/2; j++)
	{
		CkNew[j] = creal(NL[j] / M[j]) * pow(Sc1() / Sc2(), -1.5);
	}

	for ( int j = N/2 + 1; j < N; j++)
	{
		CkNew[j] = 0.0 + I * 0.0;
	}

	CkNew[N/2] = 0.0 + I * 0.0;

	for ( int j = 0; j < N; j++)
	{
		Ck[j] = CkNew[j];
	}
}

void Print_data()
{
	fftw_execute(Ck_2_Cx_);

	h = (b - a)/N;
	sprintf(Name, "//home//srdream//WORK//Breather//Find_breather//Cx//Cx_%d.txt", Iter);
	pfile = fopen(Name, "w"); 

	for (int j = 0; j < N; j++)
	{
		fprintf(pfile, "%e\t%e\n", a + j * h, cabs(Cx[j]));
	}

	fclose(pfile);

	sprintf(Name, "//home//srdream//WORK//Breather//Find_breather//Ck//Ck_%d.txt", Iter);
	pfile = fopen(Name, "w"); 

	for (int j = 0; j < N/2; j++)
	{
		fprintf(pfile, "%d\t%e\n", j, cabs(Ck[j]));
	}

	fclose(pfile);

	pfile = fopen ("Single_Breather_mu01_k0_100", "w+b");
	bin_number = fwrite(&Ck[0], sizeof(fftw_complex), N, pfile);
	fclose(pfile);

	printf("Data have been successfully written \n\n");
}


double NumberOfWaves()
{
	double Num = 0.0;

	for (int j = 1; j < N/2; j++)
	{
		Num = Num + creal((Ck[j] * conj(Ck[j]))/(j * Dk));
	}

	return Num;
}

double ErrorCalc()
{
	double Err = 0.0;
	double Num = 0.0;
	double Denum = 0.0;
	for (int j = 0; j < N; j++)
	{
		Num = Num + pow(creal(CkNew[j] - Ckini[j]), 2);
		Denum = Denum + pow(creal(CkNew[j]), 2);
	}
	Err = sqrt(Num/Denum);
	return Err;
}

void FindBreather()
{

	while (Error > MinErr)
	{
		Breather();
		Error = ErrorCalc();
		Iter++;
	}

	printf(" Iter = %d\t Err = %e\t W = %.16e\t CabsC_x = %e\n ", Iter, ErrorCalc(), W, cabs(Cx[0]));

}

void x_Shift_Breather(double x_shift)
{
	for (int j = 0; j < N/2; j++)
	{
		Ck[j] = Ck[j] * cexp(-I * j * Dk * x_shift);
	} 
}


void BashScript(int j)
{
	sprintf(Name, "BS_V0_%d.sh", j);
	pfile = fopen(Name,"w");
	sprintf(Name, "cd /mnt/scratch/ws/svdremov/202007211633Experiment_Data/2_Breathers_BS/2_Breathers_BS_%d", j);
	fprintf(pfile, "%s\n\n%s\n%s\n\n%s\n\n%s\n\n%s\n\n%s\n", "#!/bin/bash", "#PBS -l select=1:ncpus=1:mem=50m", "#PBS -l walltime=10:00:00", Name, "make", "./wave", "echo \"SCV is gotta go, sir\"" );
	fclose(pfile);
}

void Clean()
{
	system("exec rm -r //home//srdream//WORK//Breather//Find_breather//Cx/*");
	system("exec rm -r //home//srdream//WORK//Breather//Find_breather//Ck/*");

	printf("Old data were deleted \n\n");
}

////////////////////////////////////////////////////////////////////////////// MAIN


int main()
{
	srand(time(NULL));

	Set_data();

	CreatePlans();

	Clean();

	FindBreather();

	printf("\n The found breather has the following parameters: \n\n V_gr = %e, k0 = %d, Omega = %e, mu_max = %e\n", V, k0, W, pow(w0, -0.5) * sqrt(2.0) * k0 * Dk * cabs(Cx[0]));

	x_Shift_Breather((b - a)/2.0);

	Print_data();

	Closing();

	return 0;
}
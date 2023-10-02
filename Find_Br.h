	const int N = 8192;
	int k0;

	double a = 0.0;
	double b = 10000;
	double x0 = 0 * M_PI;

	//double V0 = 9.878; // k0 = 50
	double V0 = 6.24920; // k0 = 100

	double V;
	double g = 9.81;
	double Dk, h, t, tau, W, w0, delta, A;
	int Iter;

	int bin_number;

	double MinErr = 1e-15;
	double Error = 1;
	char Name[100];

	fftw_complex *Cx, *Ck, *NL, *Ckini, *M, *CkNew, *CCx, *CCk, *Cdx, *Cdk, *CCCdx, *CCCdk, *KCCk, *KCCx, *KCCCx, *KCCCk, *Ck_buff;
	fftw_complex *K, *IK;
	fftw_plan Cx_2_Ck, Ck_2_Cx_, CCx_2_CCk, Cdk_2_Cdx_, KCCk_2_KCCx_, CCCdx_2_CCCdk, KCCCx_2_KCCCk;
	FILE *pfile;


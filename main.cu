#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "convolution.cuh"

 #define USE_SHARED_MEM // Question 2 et 3
 #define USE_PITCH  // Question 3

void init_A(
	/* IN */
	noyau_t *noyau,
	int Ni,
	int Nj,
	/* OUT */
	float **A
	) {

    const float pasx = 1/float(Ni);
    const float pasy = 1/float(Nj);

    for(int i = noyau->R ; i < Ni+noyau->R ; i++) {
	    const float x = 2*M_PI*(0.5*pasx + (i-noyau->R)*pasx);

	    for(int j = noyau->R ; j < Nj+noyau->R ; j++) {
	        const float y = 2*M_PI*(0.5*pasy + (j-noyau->R)*pasy);
	        A[i][j] = sin(x)*cos(y);
	    }
    }
}

void periodic_bc(
	/* IN */
	noyau_t *noyau,
	int Ni,
	int Nj,
	/* INOUT */
	float **A
	) {

    /* WEST BOUNDARY UPDATE */
    for(int i = noyau->R ; i < Ni+noyau->R ; i++) {
	    for(int j = 0 ; j < noyau->R ; j++) {
	        A[i][j] = A[i][j+Nj];
	    }
    }

    /* EAST BOUNDARY UPDATE */
    for(int i = noyau->R ; i < Ni+noyau->R ; i++) {
	    for(int j = Nj+noyau->R ; j < Nj+2*noyau->R ; j++) {
	        A[i][j] = A[i][j-Nj];
	    }
    }

    /* SOUTH BOUNDARY UPDATE */
    for(int i = 0 ; i < noyau->R ; i++) {
	    for(int j = 0 ; j < Nj+2*noyau->R ; j++) {
	        A[i][j] = A[i+Ni][j];
	    }
    }

    /* NORTH BOUNDARY UPDATE */
    for(int i = Ni+noyau->R ; i < Ni+2*noyau->R ; i++) {
	    for(int j = 0 ; j < Nj+2*noyau->R ; j++) {
	        A[i][j] = A[i-Ni][j];
	    }
    }
}

void output(
	/* IN */
	noyau_t *noyau,
	int Ni,
	int Nj,
	float **A,
	const char *fichier
	) {
    
    FILE *fd = fopen(fichier, "w");

    const float pasx = 1/float(Ni);
    const float pasy = 1/float(Nj);

    for(int i = noyau->R ; i < Ni+noyau->R ; i++) {
	    const float x = 2*M_PI*(0.5*pasx + (i-noyau->R)*pasx);

	    for(int j = noyau->R ; j < Nj+noyau->R ; j++) {
	        const float y = 2*M_PI*(0.5*pasy + (j-noyau->R)*pasy);

	        fprintf(fd, "%.4e %.4e %.4e\n", x, y, A[i][j]);
	    }
	    fprintf(fd, "\n");
    }

    fclose(fd);
    printf("Ecriture fichier %s terminee.\n", fichier);
}

void check_noerr(int linenr) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
	    printf("Erreur ligne %d : %s \n", linenr, cudaGetErrorString(err));
	    abort();
    }
}


int main(int argc, char **argv) {
    int Ni, Nj;

    if (argc !=3) {
        printf("Usage: ./convolution Ni Nj\n");
        abort();
    }
    
    Ni = atoi(argv[1]);
    Nj = atoi(argv[2]);

    float *buf_A, *buf_B;
    float **A, **B;

    float *d_buf_A, *d_buf_B;
    size_t pitchA, pitchB;

    dim3 bThreads(DIM_BX, DIM_BY);
    dim3 gBlocs;

    gBlocs.x = (Ni + bThreads.x-1) / bThreads.x;
    gBlocs.y = (Nj + bThreads.y-1) / bThreads.y;

    printf("nthreads : (%d, %d)\n",  bThreads.x, bThreads.y);
    printf("nblocks : (%d, %d)\n",  gBlocs.x, gBlocs.y);
    //printf("no erreur\n");

    int Ni_2R, Nj_2R;
    //printf("no erreur");
    noyau_t noyau;
    //printf("no erreur");
    noyau_t *d_noyau; // pointeur noyau sur device
    

    //Question1
    //d_noyau = init_1thread_noyau(&noyau, R_NOYAU);
    //printf("no erreur");

    //Question2
    d_noyau = init_noyau(&noyau, R_NOYAU);
    //printf("no erreur");

    Ni_2R = Ni + 2*noyau.R;
    Nj_2R = Nj + 2*noyau.R;

    buf_A = (float*)malloc(Ni_2R*Nj_2R*sizeof(float));
    buf_B = (float*)malloc(Ni_2R*Nj_2R*sizeof(float));

    A = (float**)malloc(Ni_2R*sizeof(float*));
    B = (float**)malloc(Ni_2R*sizeof(float*));

    for(int i = 0 ; i < Ni_2R ; i++) {
	    A[i] = buf_A + i*Nj_2R;
	    B[i] = buf_B + i*Nj_2R;
    }

    //afficher_matrice(d_noyau, R_NOYAU);

//Question4
#ifdef USE_PITCH
    // A COMPLETER
    d_buf_A = NULL;
    d_buf_B = NULL;
    cudaMallocPitch(&d_buf_A, &pitchA, (sizeof(int)*Ni_2R), Nj_2R);
    cudaMallocPitch(&d_buf_B, &pitchB, (sizeof(int)*Ni_2R), Nj_2R);

#else
    cudaMalloc(&d_buf_A, Ni_2R*Nj_2R*sizeof(float));
    cudaMalloc(&d_buf_B, Ni_2R*Nj_2R*sizeof(float));

    pitchA = Nj_2R*sizeof(float);
    pitchB = Nj_2R*sizeof(float);
#endif

    init_A(&noyau, Ni, Nj, A);
    output(&noyau, Ni, Nj, A, "A000");

    for(int iter = 1 ; iter <= 5 ; iter++) {
	    periodic_bc(&noyau, Ni, Nj, A);

	// Transfert CPU => GPU
//Question4
#ifdef USE_PITCH
	// A COMPLETER
	cudaMemcpy2D(d_buf_A, pitchA, buf_A, (sizeof(float)*Ni_2R), (sizeof(float)*Ni_2R), Nj_2R, cudaMemcpyHostToDevice);

#else
	    cudaMemcpy(
		    d_buf_A, 
		    buf_A, 
		    Ni_2R*Nj_2R*sizeof(float), 
		    cudaMemcpyHostToDevice);
#endif
	    check_noerr(__LINE__);

	// Application operateur convolution
//Question3
#ifdef USE_SHARED_MEM
	    convol_sh<<<gBlocs, bThreads>>>(
		    d_noyau, 
		    d_buf_A, pitchA,
		    Ni, Nj, 
		    d_buf_B, pitchB);
#else
	    convol_gl<<<gBlocs, bThreads>>>(
		    d_noyau, 
		    d_buf_A, pitchA,
		    Ni, Nj, 
		    d_buf_B, pitchB);
#endif
	    check_noerr(__LINE__);

	// Transfert GPU => CPU
//Question4
#ifdef USE_PITCH
	// A COMPLETER
	cudaMemcpy2D(buf_B, (sizeof(float)*Ni_2R), d_buf_B, pitchB, (sizeof(float)*Ni_2R), Nj_2R, cudaMemcpyDeviceToHost);
#else
	    cudaMemcpy(
		    buf_B,
		    d_buf_B, 
		    Ni_2R*Nj_2R*sizeof(float), 
		    cudaMemcpyDeviceToHost);
#endif
	    check_noerr(__LINE__);

	    float *buf_C;
	    buf_C = buf_A;
	    buf_A = buf_B;
	    buf_B = buf_C;

	    float **C;
	    C = A;
	    A = B;
	    B = C;
    }

    output(&noyau, Ni, Nj, A, "A001");

    free(buf_A);
    free(buf_B);
    free(A);
    free(B);

    free_noyau(&noyau, d_noyau);

    return 0;
}


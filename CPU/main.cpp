#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "convolution.h"

const int R = 2;

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

	    for(int j = noyau->R ; j < Nj+noyau->R ; j++){
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

    int Ni_2R, Nj_2R;
    noyau_t noyau;

    init_noyau(&noyau, R);

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

    init_A(&noyau, Ni, Nj, A);
    output(&noyau, Ni, Nj, A, "A000");

    for(int iter = 1 ; iter <= 5 ; iter++) {
	    periodic_bc(&noyau, Ni, Nj, A);
	    convolution(&noyau, A, Ni, Nj, B);

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

    free_noyau(&noyau);

    return 0;
}

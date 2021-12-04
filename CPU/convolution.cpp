#include <stdlib.h>
#include <stdio.h>

#include "convolution.h"

void init_noyau(noyau_t *noyau, int R)
{
    noyau->R = R;
    noyau->KMAX = (2*R + 1)*(2*R + 1);

    int k = 0;
    float coeff = -1/float(noyau->KMAX);

    noyau->indi =   (int*)malloc(noyau->KMAX * sizeof(int));
    noyau->indj =   (int*)malloc(noyau->KMAX * sizeof(int));
    noyau->C    = (float*)malloc(noyau->KMAX * sizeof(float));

    for(int i = -R ; i <= +R ; i++)
    {
	for(int j = -R ; j <= +R ; j++)
	{
	    noyau->indi[k] = i;
	    noyau->indj[k] = j;
	    noyau->C[k] = ((i < R && j < R) ? 1 : 0) * coeff;
	    k++;
	}
    }
    noyau->C[R*(2*R+1)+R] = 1;
    k = 0;
}

void free_noyau(noyau_t *noyau)
{
    free(noyau->indi);
    free(noyau->indj);
    free(noyau->C);
}

void convolution(
	noyau_t *noyau,
	/* IN */
	float **A, 
	int Ni, 
	int Nj,
	/* OUT */
	float **B
	)
{
    for(int i = noyau->R ; i < Ni+noyau->R ; i++)
    {
	for(int j = noyau->R ; j < Nj+noyau->R ; j++)
	{
	    B[i][j] = 0;

	    for(int k = 0 ; k < noyau->KMAX ; k++)
	    {
		const float val_A = A[i+noyau->indi[k]][j+noyau->indj[k]];

		B[i][j] += noyau->C[k] * val_A;
	    }
	}
    }
}


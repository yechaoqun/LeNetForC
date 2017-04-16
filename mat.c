#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mat.h"
/*
 *			-----------
 *                | x-1, y-1 | x, y-1 | x+1, y-1 |
 *                | x-1, y    | x, y   -| x+1, y    |
 *                | x-1, y+1 | x, y+1 | x+1, y+1 |
 *
 */

static inline int max(int a, int b, int c)
{
	if(a < b)
		a = b;
	if(a < c)
		a = c;

	return a;
}

static inline int min(int a, int b, int c)
{
	if(a > b)
		a = b;
	if(a > c)
		a = c;

	return a;
}

void matPrint(const char *name, float *mat, int row, int col)
{
#if 0
	int x, y;
printf("[%s]\n", name);
	for(x=0; x<row; x++)
	{
		for(y=0; y<col; y++)
			printf("%3.3f, ", *(mat+x*row+y));
		printf("\n");
	}
printf("==============\n");
#endif
}

int matAve3x3(int m1, int m2, int m3, int m4, int m5, int m6, int m7, int m8, int m9)
{
	return 0;
}



int matMaxDV3x3(int m1, int m2, int m3, int m4, int m5, int m6, int m7, int m8, int m9)
{
	int ma1, ma2, ma3;
	int mi1, mi2, mi3;

	ma1 = max(m1, m2, m3);
	ma2 = max(m4, m5, m6);
	ma3 = max(m7, m8, m9);

	mi1 = min(m1, m2, m3);
	mi2 = min(m4, m5, m6);
	mi3 = min(m7, m8, m9);

	return 0;
//	return max(ma1, ma2, ma3) - min(mi1, mi2, mi3);
}

float matConv(float *mat1, float *mat2, int row, int col)
{
	int x,y;
	float t = 0;
	float a, b;

	for(x=0; x<row; x++)
	{
		for(y=0; y<col; y++)
		{
			a = *(mat1+x*row+y);
			b = *(mat2+x*row+y);
			t += a*b;
		}
	}

	return t;
}

float matSum(float *mat, int row, int col)
{
	float t = 0;
	int x, y;
	
	for(x=0; x<row; x++)
		for(y=0; y<col; y++)
		{
			t += *(mat+x*row+y);
		}

	return t;
}

void matConvFactor(float *mat, float *mat2, int row, int col, float factor)
{
	int x, y;

	for(x=0; x<row; x++)
		for(y=0; y<col; y++)
		{
			*(mat+x*row+y) = (*(mat2+x*row+y))*factor;
		}
}

void matFitConvFactor(float *out_mat, int out_row, int out_col, float *mat2, int row, int col, float factor)
{
	int x, y;

	for(x=0; x<row; x++)
		for(y=0; y<col; y++)
		{
			*(out_mat+x*out_row+y) = *(mat2+x*row+y)*factor;
		}
}

void matAdd(float *mat1, float *mat2, int row, int col)
{
	int x, y;

	for(x=0; x<row; x++)
		for(y=0; y<col; y++)
		{
			*(mat1+x*row+y) += *(mat2+x*row+y);
		}
}

void matFitAdd(float *out_mat, int out_row, int out_col, float *mat2, int row, int col)
{
	int x, y;

	for(x=0; x<row; x++)
		for(y=0; y<col; y++)
		{
			*(out_mat+x*out_row+y) += *(mat2+x*row+y);
		}
}

void matAddByFactor(float *mat, float *mat2, int row, int col, float factor)
{
	int x, y;

	for(x=0; x<row; x++)
		for(y=0; y<col; y++)
		{
			*(mat+x*row+y) += *(mat2+x*row+y)*factor;
		}
}

void upSample(float *out, int outsize, float *input, int insize, int factor)
{
	int x, y;
	
	for(x=0; x<outsize; x++)
	{
		for(y=0; y<outsize; y++)
		{
			*(out+x*outsize+y) = *(input+(x/factor)*insize+y/factor) / (factor*factor);
		}
	}
}




#ifndef __MAT_H__
#define __MAT_H__










void matPrint(const char * name,float * mat,int row,int col);
float matConv(float * mat1,float * mat2,int row,int col);
float matSum(float * mat,int row,int col);
void matConvFactor(float * mat,float * mat2,int row,int col, float factor);
void matAdd(float * mat1,float * mat2,int row,int col);
void matFitAdd(float * out_mat,int out_row,int out_col,float * mat2,int row,int col);
void matAddByFactor(float * mat,float * mat2,int row,int col, float factor);
void matFitConvFactor(float *out_mat, int out_row, int out_col, float *mat2, int row, int col, float factor);
void upSample(float * out,int outsize,float * input,int insize, int factor);




#endif


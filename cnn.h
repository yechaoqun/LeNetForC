#ifndef __CNN_H__
#define __CNN_H__


typedef struct stC1Cell {
	float W[5][5];
	float basis;

	float dw[28][28];
	float db;
}C1Cell;


typedef struct stC1Layer {
	C1Cell cell[6];

	float x[32][32]; //data input
}C1Layer;

typedef struct stS2Cell {
	float c1in[28][28];	//c1 input
	float s2out[14][14];  //s2 output

	float dw[14][14];
}S2Cell;


typedef struct stS2Layer {
	S2Cell cell[6];
}S2Layer;

typedef struct stC3Cell {
	float W[5][5];
	float basis;

	float dw[10][10];
}C3Cell;

typedef struct stC3Layer {
	C3Cell cell[16];
}C3Layer;

typedef struct stS4Cell {
	float c3in[10][10]; //c3 input
	float s4out[5][5];	//s4 output

	float dw[5][5];
}S4Cell;

typedef struct stS4Layer {
	
	S4Cell cell[16];
}S4Layer;

typedef struct stO5Cell {
	float W[400];
	float basis;

	float out;
	float delta;
}O5Cell;

typedef struct stO5Layer {
	O5Cell cell[10];

	float vector[400];	//s4输出展开为1维向量
}O5Layer;

typedef struct stLeNet {
	C1Layer c1;
	S2Layer s2;
	C3Layer c3;
	S4Layer s4;
	O5Layer o5;
}LeNet;

typedef struct stMnistImage {
	float img[32][32];
	float lable;
}MnistImage;

typedef struct stMnist {
	int train_samples;
	int test_samples;
	
	MnistImage *train_data;
	MnistImage *test_data;
}Mnist;

#endif


/*
 *LeNet 的C语言版本实现
 *
 *作者:yechaoqun
 *mail	:yechaoqun1006@163.com
 *
 *
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>

#include "cnn.h"
#include "mat.h"

#define DEBUG	0

#define MNIST_TRAIN_IMAGE	("train-images.idx3-ubyte")
#define MNIST_TRAIN_LABLE	("train-labels.idx1-ubyte")
#define MNIST_TEST_IMAGE	("t10k-images.idx3-ubyte")
#define MNIST_TEST_LABLE	("t10k-labels.idx1-ubyte")

#if DEBUG
static char con_map1[16][6] = 
{
	{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0},
	{0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1},
	{0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 1},
	{0, 0, 0, 0, 0, 1}, {0, 1, 1, 0, 1, 1}, {0, 0, 1, 1, 0, 1}, {0, 1, 1, 1, 1, 1}
};

#else
static char con_map1[16][6] = 
{
	{1, 1, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 1, 1},
	{1, 0, 0, 0, 1, 1}, {1, 1, 0, 0, 0, 1}, {1, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 1, 0},
	{0, 0, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 1, 0, 0, 1, 1}, {1, 1, 1, 0, 0, 1},
	{1, 1, 0, 1, 1, 0}, {0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 1}
};
#endif

static char con_map2[6][16];

int reverse32(int i)   
{  
	unsigned char ch1, ch2, ch3, ch4;  
	ch1 = i & 255;  
	ch2 = (i >> 8) & 255;  
	ch3 = (i >> 16) & 255;  
	ch4 = (i >> 24) & 255;  
	return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;  
}

int loadMnist(Mnist *mnist)
{
	int fd = -1;
	int num, magic, col, row;
	int i,x,y;
	unsigned char temp;

	/*1.read image data*/
	fd = open(MNIST_TRAIN_IMAGE, O_RDONLY);
	if(fd == -1)
	{
		printf("load %s failed\n", MNIST_TRAIN_IMAGE);
		return -1;
	}

	read(fd, &magic, 4);
	magic = reverse32(magic);

	read(fd, &num, 4);
	num = reverse32(num);

	mnist->train_samples = num;
	mnist->train_data = malloc(num*sizeof(MnistImage));
	memset(mnist->train_data, 0, num*sizeof(MnistImage));

	read(fd, &row, 4);
	read(fd, &col, 4);
	row = reverse32(row);
	col = reverse32(col);
printf("train data:%d, %d, %d\n", num, row, col);	
#if 1
	for(i=0; i<num; i++)
	{
		for(x=2; x<row+2; x++)
		{
			for(y=2; y<col+2; y++)
			{
				read(fd, &temp, 1);
				mnist->train_data[i].img[x][y] = (float)(temp);
			}
		}
	}
#endif
	close(fd);

	/*2.read lable*/
	fd = open(MNIST_TRAIN_LABLE, O_RDONLY);
	if(fd == -1)
	{
		printf("load %s failed\n", MNIST_TRAIN_LABLE);
		return -1;
	}

	read(fd, &magic, 4);
	magic = reverse32(magic);

	read(fd, &num, 4);
	num = reverse32(num);

printf("train lable:%d\n", num);
#if 1
	for(i=0; i<num; i++)
	{
		read(fd, &temp, 1);
		mnist->train_data[i].lable = temp;
	}
#endif
	close(fd);

	/*1.read image data*/
	fd = open(MNIST_TEST_IMAGE, O_RDONLY);
	if(fd == -1)
	{
		printf("load %s failed\n", MNIST_TEST_IMAGE);
		return -1;
	}

	read(fd, &magic, 4);
	magic = reverse32(magic);

	read(fd, &num, 4);
	num = reverse32(num);

	mnist->test_samples = num;
	mnist->test_data = malloc(num*sizeof(MnistImage));
	memset(mnist->test_data, 0, num*sizeof(MnistImage));

	read(fd, &row, 4);
	read(fd, &col, 4);
	row = reverse32(row);
	col = reverse32(col);
printf("test data:%d, %d, %d\n", num, row, col);	
#if 1
	for(i=0; i<num; i++)
	{
		for(x=2; x<row+2; x++)
		{
			for(y=2; y<col+2; y++)
			{
				read(fd, &temp, 1);
				mnist->test_data[i].img[x][y] = (float)(temp);
			}
		}
	}
#endif
	close(fd);

	
	/*2.read lable*/
	fd = open(MNIST_TEST_LABLE, O_RDONLY);
	if(fd == -1)
	{
		printf("load %s failed\n", MNIST_TEST_LABLE);
		return -1;
	}

	read(fd, &magic, 4);
	magic = reverse32(magic);

	read(fd, &num, 4);
	num = reverse32(num);

printf("test lable:%d\n", num);
#if 1
	for(i=0; i<num; i++)
	{
		read(fd, &temp, 1);
		mnist->test_data[i].lable = temp;
	}
#endif
	close(fd);

	return 0;
}

void saveMnistStatic(Mnist *mnist)
{
	int fd = -1;

	fd = open("mnist.bin", O_CREAT | O_TRUNC | O_RDWR, 666);
	write(fd, &(mnist->train_samples), sizeof(mnist->train_samples));
	write(fd, &(mnist->test_samples), sizeof(mnist->test_samples));
	write(fd, (char*)mnist->train_data, mnist->train_samples*sizeof(MnistImage));
	write(fd, (char*)mnist->test_data, mnist->test_samples*sizeof(MnistImage));
	close(fd);
}

void loadMnistStatic(Mnist * mnist)
{
	int fd = open("mnist.bin", O_RDONLY);
	
	read(fd, &(mnist->train_samples), sizeof(mnist->train_samples));
	read(fd, &(mnist->test_samples), sizeof(mnist->test_samples));

	mnist->train_data = malloc(mnist->train_samples*sizeof(MnistImage));
	mnist->test_data  = malloc(mnist->test_samples*sizeof(MnistImage));

	read(fd, (char *)mnist->train_data, mnist->train_samples*sizeof(MnistImage));
	read(fd, (char *)mnist->test_data, mnist->test_samples*sizeof(MnistImage));
}

#if 0

void loadBmpMnist(Mnist *mnist)
{
	int i = 0, oft;
	int fd = -1;
	char name[30];
	BITMAP_FILE rgb;
	HSL_FILE hsl;

	mnist->test_data = malloc(10*sizeof(MnistImage));
	mnist->test_samples = 0;

	for(i=0; i<10; i++)
	{
		sprintf(name, "./bmp/%d.bmp", i);
		fd = open(name, O_RDONLY);
		if(fd == -1)
		{
			printf("open failed\n");
			return;
		}

		readBMPFile(fd, &rgb);
		close(fd);
		RGB2HSV(&hsl, &rgb);
		
		for(oft=0; oft<32*32; oft++)
		{
			mnist->test_data[i].img[oft/32][(32-oft)%32] = hsl.buffer[3*32*32-3*oft];
		}
		mnist->test_data[i].lable = i;
	}
	mnist->test_samples = 10;
}

void savebmp(Mnist *mnist)
{
	int fd = -1;
	int i = 0, oft = 0;
	char fname[200];
	BITMAP_FILE src, dst;

	fd = open("0.bmp", O_RDONLY);
	readBMPFile(fd, &src);
	close(fd);

	for(i=59990; i<mnist->train_samples; i++)
	{
		sprintf(fname, "t%02d_%04d.bmp", i, (int)(mnist->train_data[i].lable));
		for(oft=0; oft<32*32; oft++)
		{
			src.buffer[3*32*32-3*oft]   = mnist->train_data[i].img[oft/32][(32-oft)%32];
			src.buffer[3*32*32-(3*oft+1)] = src.buffer[3*32*32-(3*oft+2)] = src.buffer[3*32*32-3*oft];
		}

		
		fd = open(fname, O_CREAT | O_TRUNC | O_RDWR, 666);
		writeBMPFile(fd, &src);
		close(fd);
	}

	for(i=9990; i<mnist->test_samples; i++)
	{
		sprintf(fname, "d%02d_%04d.bmp", i, 0);
		for(oft=0; oft<32*32; oft++)
		{
			src.buffer[3*32*32-3*oft]   = mnist->test_data[i].img[oft/32][(32-oft)%32];
			src.buffer[3*32*32-(3*oft+1)] = src.buffer[3*32*32-(3*oft+2)] = src.buffer[3*32*32-3*oft];
		}

		
		fd = open(fname, O_CREAT | O_TRUNC | O_RDWR, 666);
		writeBMPFile(fd, &src);
		close(fd);
	}
}
#endif

void preprocessMnist(Mnist *mnist)
{
	int i = 0;
	int x, y;
	
	for(i=0; i<mnist->train_samples; i++)
	{
		for(x=0; x<32; x++)
		{
			for(y=0; y<32; y++)
			{
				mnist->train_data[i].img[x][y] = mnist->train_data[i].img[x][y]/255.0;
			}
		}
	}

	for(i=0; i<mnist->test_samples; i++)
	{
		for(x=0; x<32; x++)
		{
			for(y=0; y<32; y++)
			{
				mnist->test_data[i].img[x][y] = mnist->test_data[i].img[x][y]/255.0;
			}
		}
	}
}


void initC1(C1Layer *c1)
{
	int idx, x, y;
	float randnum = 0;

	srand((unsigned)time(NULL));
	for(idx=0; idx<6; idx++)
	{
		for(x=0; x<5; x++)
			for(y=0; y<5; y++)
			{
			#if DEBUG
				c1->cell[idx].W[x][y] = 0;
				c1->cell[idx].W[0][0] = 1;
			#else
				randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2; 
				c1->cell[idx].W[x][y] = randnum*sqrt((float)6.0/(float)(5*5*(1+6)));
			#endif
			}
	}
}

void initS2(S2Layer *s2)
{
}

void initC3(C3Layer *c3)
{
	int idx, x, y;
	float randnum = 0;

	srand((unsigned)time(NULL));
	for(idx=0; idx<16; idx++)
	{
		for(x=0; x<5; x++)
			for(y=0; y<5; y++)
			{
			#if DEBUG
				c3->cell[idx].W[x][y] = 0;
				c3->cell[idx].W[4][4] = 1;
			#else
				randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2; 
				c3->cell[idx].W[x][y] = randnum*sqrt((float)6.0/(float)(5*5*(6+16)));
			#endif
			}
	}
}

void initS4(S4Layer *s4)
{
}

void initO5(O5Layer *o5)
{
	int idx, x, y;
	float randnum = 0;

	srand((unsigned)time(NULL));
	for(idx=0; idx<10; idx++)
	{
		for(x=0; x<400; x++)
		{
		#if DEBUG
			o5->cell[idx].W[x] = 1;
		#else
			randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2; 
			o5->cell[idx].W[x] = randnum*sqrt((float)6.0/(float)(400+10));
		#endif
		}
	}
}

void saveNet(LeNet *net)
{
	int i, len = 0;
	int fd = -1;

	fd = open("data.bin", O_CREAT | O_TRUNC | O_RDWR, 666);
	if(fd == -1)
	{
		printf("open data.bin failed\n");
		return;
	}

	for(i=0; i<6; i++)
	{
		len += write(fd, net->c1.cell[i].W, sizeof(net->c1.cell[i].W));
	}

	for(i=0; i<16; i++)
	{
		len += write(fd, net->c3.cell[i].W, sizeof(net->c3.cell[i].W));
	}

	for(i=0; i<10; i++)
	{
		len += write(fd, net->o5.cell[i].W, sizeof(net->o5.cell[i].W));
	}

	close(fd);

	if(len != (6*sizeof(net->c1.cell[i].W) + 16*sizeof(net->c3.cell[i].W) + 10*sizeof(net->o5.cell[i].W)))
	{
		printf("write data.bin failed\n");
	}
}

void loadNet(LeNet *net)
{
	int i, len = 0;
	int fd = -1;

	fd = open("data.bin", O_RDONLY);
	if(fd == -1)
	{
		printf("open data.bin failed\n");
		return;
	}

	for(i=0; i<6; i++)
	{
		len += read(fd, net->c1.cell[i].W, sizeof(net->c1.cell[i].W));
	}

	for(i=0; i<16; i++)
	{
		len += read(fd, net->c3.cell[i].W, sizeof(net->c3.cell[i].W));
	}

	for(i=0; i<10; i++)
	{
		len += read(fd, net->o5.cell[i].W, sizeof(net->o5.cell[i].W));
	}

	close(fd);

	if(len != (6*sizeof(net->c1.cell[i].W) + 16*sizeof(net->c3.cell[i].W) + 10*sizeof(net->o5.cell[i].W)))
	{
		printf("read data.bin failed\n");
	}
}

void backupNet(LeNet *net, LeNet *backup)
{
	memcpy(backup, net, sizeof(LeNet));
}

void prepareLeNet(LeNet *net)
{
	int i, k, x, y;

	for(i=0; i<6; i++)
	{
		for(k=0; k<16; k++)
		{
			if(con_map1[k][i])
			{
				con_map2[i][k] = 1;
			}
			else
			{
				con_map2[i][k] = 0;
			}
		}
	}

#if 1
	initC1(&net->c1);
	initS2(&net->s2);
	initC3(&net->c3);
	initS4(&net->s4);
	initO5(&net->o5);
#endif
}

float sigmod(float x)
{
	return (float)1.0/((float)(1.0+exp(-x)));
//	return x*(x>0);
}

float antiSigmod(float x)
{
	return x*(1-x);
	//return x>0;
}



void stepC1(LeNet *net, MnistImage *in)
{
	int cidx = 0;
	int x,y;
	float basis, tmp;
	C1Layer *c1 = &net->c1;
	float ma[5][5];

	for(cidx=0; cidx<6; cidx++)
	{
		//WX+b
		tmp = 0;
		basis = net->c1.cell[cidx].basis;
		for(x=0; x<28; x++)
		{
			for(y=0; y<28; y++)
			{
				memcpy(&ma[0][0], &(in->img[x][y]), 5*sizeof(float));
				memcpy(&ma[1][0], &(in->img[x+1][y]), 5*sizeof(float));
				memcpy(&ma[2][0], &(in->img[x+2][y]), 5*sizeof(float));
				memcpy(&ma[3][0], &(in->img[x+3][y]), 5*sizeof(float));
				memcpy(&ma[4][0], &(in->img[x+4][y]), 5*sizeof(float));
			
				tmp = matConv((float*)net->c1.cell[cidx].W, (float*)ma, 5, 5)+basis;
				net->s2.cell[cidx].c1in[x][y] = sigmod(tmp);
			}
		}
	}

	memcpy(c1->x, in->img, 32*32*sizeof(float));
}

void stepS2(LeNet *net)
{
	int sidx = 0;
	int x, y;
	float ave[2][2] = {{0.25,0.25}, {0.25,0.25}};
	float ma[2][2];

	for(sidx=0; sidx<6; sidx++)
	{
		for(x=0; x<14; x++)
		{
			for(y=0; y<14; y++)
			{
				ma[0][0] = net->s2.cell[sidx].c1in[2*x][2*y];
				ma[0][1] = net->s2.cell[sidx].c1in[2*x][2*y+1];
				ma[1][0] = net->s2.cell[sidx].c1in[2*x+1][2*y];
				ma[1][1] = net->s2.cell[sidx].c1in[2*x+1][2*y+1];
				net->s2.cell[sidx].s2out[x][y] = matConv((float *)ma, (float*)ave, 2, 2);
			}
		}
	}
}

void stepC3(LeNet *net)
{
	int cidx = 0;
	int x,y,k;
	float basis, tmp;
	C3Layer *c3 = &net->c3;
	float ma[5][5];
	

	for(cidx=0; cidx<16; cidx++)
	{
		//WX+b
		basis = net->c3.cell[cidx].basis;
		for(x=0; x<10; x++)
		{
			for(y=0; y<10; y++)
			{
				tmp = 0;
				for(k=0; k<6; k++)
				{
					if(con_map1[cidx][k])
					{
						memcpy(&ma[0][0], &(net->s2.cell[k].s2out[x][y]),   5*sizeof(float));
						memcpy(&ma[1][0], &(net->s2.cell[k].s2out[x+1][y]), 5*sizeof(float));
						memcpy(&ma[2][0], &(net->s2.cell[k].s2out[x+2][y]), 5*sizeof(float));
						memcpy(&ma[3][0], &(net->s2.cell[k].s2out[x+3][y]), 5*sizeof(float));
						memcpy(&ma[4][0], &(net->s2.cell[k].s2out[x+4][y]), 5*sizeof(float));
						tmp += matConv((float*)net->c3.cell[cidx].W, (float *)ma, 5, 5);
					}
				}
				net->s4.cell[cidx].c3in[x][y] = sigmod(tmp+basis);
			}
		}
	}

}

void stepS4(LeNet *net)
{
	int sidx = 0;
	int x, y;
	float ave[2][2] = {{0.25,0.25}, {0.25,0.25}};
	float ma[2][2];

	for(sidx=0; sidx<16; sidx++)
	{
		for(x=0; x<5; x++)
		{
			for(y=0; y<5; y++)
			{
				ma[0][0] = net->s4.cell[sidx].c3in[2*x][2*y];
				ma[0][1] = net->s4.cell[sidx].c3in[2*x][2*y+1];
				ma[1][0] = net->s4.cell[sidx].c3in[2*x+1][2*y];
				ma[1][1] = net->s4.cell[sidx].c3in[2*x+1][2*y+1];
				net->s4.cell[sidx].s4out[x][y] = matConv((float *)ma, (float*)ave, 2, 2);
			}
		}
	}
}

void stepSpread(LeNet *net)
{
	int i = 0;
	for(i=0; i<16; i++)
	{
		memcpy(&net->o5.vector[25*i], net->s4.cell[i].s4out, 25*sizeof(float));
	}
}

void stepO5(LeNet *net)
{
	int i = 0;
	float tmp;
	
	for(i=0; i<10; i++)
	{
		tmp = matConv(net->o5.vector, net->o5.cell[i].W, 1, 400);
		net->o5.cell[i].out = sigmod(tmp + net->o5.cell[i].basis);
	}
}

void forward(LeNet *net, MnistImage *image)
{
	stepC1(net, image);
	stepS2(net);
	stepC3(net);
	stepS4(net);
	stepSpread(net);
	stepO5(net);
}

void backO5(LeNet *net, MnistImage *in)
{
	int i = 0;
	float e = 0;
	float tvector[10];
	
	memset(tvector, 0, sizeof(tvector));
	tvector[(int)in->lable] = 1.0;

	for(i=0; i<10; i++)
	{
		e  = net->o5.cell[i].out - tvector[i];
		net->o5.cell[i].delta = e*antiSigmod(net->o5.cell[i].out);
	}
}

void backS4(LeNet *net)
{
	int i = 0, j = 0, x, y;
	
	for(i=0; i<16; i++)
	{
		for(x=0; x<5; x++)
		{
			for(y=0; y<5; y++)
			{
				for(j=0; j<10; j++)
				{
					net->s4.cell[i].dw[x][y] += net->o5.cell[j].W[i*25+x*5+y]*net->o5.cell[j].delta;
				}
			}
		}
	}
}

void backC3(LeNet *net)
{
	int i = 0, x, y;
	float delta[10][10];

	for(i=0; i<16; i++)
	{
		upSample((float*)delta, 10, (float*)net->s4.cell[i].dw, 5, 2);
		for(x=0; x<10; x++)
		{
			for(y=0; y<10; y++)
			{
				net->c3.cell[i].dw[x][y] += delta[x][y]*antiSigmod(net->s4.cell[i].c3in[x][y]);
			}
		}
	}
}

void backS2(LeNet *net)
{
	int i, k, x, y;
	
	float btmp[16][14][14];
	float ma[5][5];

	memset(btmp, 0, sizeof(btmp));
	for(i=0; i<16; i++)
	{
		for(x=0; x<10; x++)
		{
			for(y=0; y<10; y++)
			{
				matConvFactor((float*)ma, (float*)net->c3.cell[i].W, 5, 5, net->c3.cell[i].dw[x][y]);
				matFitAdd((float*)&(btmp[i][x][y]), 14, 14, (float*)ma, 5, 5);
			}
		}
	}

	
	for(i=0; i<6; i++)
	{
		for(k=0; k<16; k++)
		{
			if(con_map2[i][k])
			{			
				for(x=0; x<14; x++)
					for(y=0; y<14; y++)
					{
						net->s2.cell[i].dw[x][y] += btmp[k][x][y];
					}
			}
		}
	}
}

void backC1(LeNet *net)
{
	int i = 0, x, y;
	float delta[28][28];

	for(i=0; i<6; i++)
	{
		upSample((float*)delta, 28, (float*)net->s2.cell[i].dw, 14, 2);
		for(x=0; x<28; x++)
		{
			for(y=0; y<28; y++)
			{
				net->c1.cell[i].dw[x][y] += delta[x][y]*antiSigmod(net->s2.cell[i].c1in[x][y]);
			}
		}
	}
}


void backward(LeNet *net, MnistImage *image)
{
	backO5(net, image);
	backS4(net);
	backC3(net);
	backS2(net);
	backC1(net);
}

void updateWB(LeNet *net)
{
	int idx, i, x, y;
	float alpha = 1;
	float dw[5][5];
	float c3in[14][14];
	float o5tmp[400];
	float ma[5][5];
	//C1
	for(idx=0; idx<6; idx++)
	{
		memset(dw, 0, sizeof(dw));
		for(x=0; x<28; x++)
			for(y=0; y<28; y++)
			{
				memcpy(&ma[0][0], &net->c1.x[x][y],   5*sizeof(float));
				memcpy(&ma[1][0], &net->c1.x[x+1][y], 5*sizeof(float));
				memcpy(&ma[2][0], &net->c1.x[x+2][y], 5*sizeof(float));
				memcpy(&ma[3][0], &net->c1.x[x+3][y], 5*sizeof(float));
				memcpy(&ma[4][0], &net->c1.x[x+4][y], 5*sizeof(float));
				matAddByFactor((float*)dw, (float*)ma, 5, 5, net->c1.cell[idx].dw[x][y]);
			}
		matConvFactor((float*)dw, (float*)dw, 5, 5, -1*alpha);
		matAdd((float*)net->c1.cell[idx].W, (float*)dw, 5, 5);
		net->c1.cell[idx].basis += -1*alpha*matSum((float*)net->c1.cell[idx].dw, 28, 28);
	}

	//C3
	for(idx=0; idx<16; idx++)
	{
		memset(dw, 0, sizeof(dw));
		memset(c3in, 0, sizeof(c3in));
		for(i=0; i<6; i++)
		{
			if(con_map1[idx][i])
			{
				matAdd((float*)c3in, (float*)net->s2.cell[i].s2out, 14, 14);
			}
		}
		for(x=0; x<10; x++)
		{
			for(y=0; y<10; y++)
			{
				memcpy(&ma[0][0], &c3in[x][y],   5*sizeof(float));
				memcpy(&ma[1][0], &c3in[x+1][y], 5*sizeof(float));
				memcpy(&ma[2][0], &c3in[x+2][y], 5*sizeof(float));
				memcpy(&ma[3][0], &c3in[x+3][y], 5*sizeof(float));
				memcpy(&ma[4][0], &c3in[x+4][y], 5*sizeof(float));
				
				matAddByFactor((float*)dw, (float*)ma, 5, 5, net->c3.cell[idx].dw[x][y]);
			}
		}
		matConvFactor((float*)dw, (float*)dw, 5, 5, -1*alpha);
		matAdd((float*)net->c3.cell[idx].W, (float*)dw, 5, 5);
		net->c3.cell[idx].basis += -1*alpha*matSum((float*)net->c3.cell[idx].dw, 10, 10);
	}

	//O5
	for(idx=0; idx<10; idx++)
	{
		matConvFactor((float*)o5tmp, (float*)net->o5.vector, 1, 400, -1*alpha*net->o5.cell[idx].delta); //E/W
		matAdd((float*)net->o5.cell[idx].W, (float*)o5tmp, 1, 400);
		net->o5.cell[idx].basis += -1*alpha*net->o5.cell[idx].delta;
		
	}
}

void clear(LeNet *net)
{
	int i = 0;

	for(i=0; i<6; i++)
	{
		memset(net->c1.cell[i].dw, 0, sizeof(net->c1.cell[i].dw));
	}

	for(i=0; i<6; i++)
	{
		memset(net->s2.cell[i].dw, 0, sizeof(net->s2.cell[i].dw));
	}

	for(i=0; i<16; i++)
	{
		memset(net->c3.cell[i].dw, 0, sizeof(net->c3.cell[i].dw));
	}

	for(i=0; i<16; i++)
	{
		memset(net->s4.cell[i].dw, 0, sizeof(net->s4.cell[i].dw));
	}

}



void train(LeNet *net, Mnist *mnist)
{
	int num = mnist->train_samples;
	int idx = 0;

	for(idx=0; idx<num; idx++)
	{
		forward(net, &mnist->train_data[idx]);
		backward(net, &mnist->train_data[idx]);		
		updateWB(net);
		clear(net);
	}
	printf("Done %d\n", num);
}

void train_batch(LeNet *net, Mnist *mnist, int start, int batch_size)
{
	int num = batch_size;
	int idx = 0;

	for(idx=0; idx<num; idx++)
	{
		forward(net, &mnist->train_data[start+idx]);
		backward(net, &mnist->train_data[start+idx]); 	
		updateWB(net);
		clear(net);
	}
	printf("Done %d\n", num);
}


float predict(LeNet *net, Mnist *mnist)
{
	int i = 0;
	int num = mnist->test_samples;
	int idx = 0, maxIdx = 0;
	float maxnum = -1, corr = 0;
	

	for(idx=0; idx<num; idx++)
	{
		maxnum = -1;
		maxIdx = 0;
		
		
		forward(net, &mnist->test_data[idx]);
		for(i=0; i<10; i++)
		{
			if(maxnum < net->o5.cell[i].out)
			{
				maxnum = net->o5.cell[i].out;
				maxIdx = i;
			}
		}

		if(maxIdx == mnist->test_data[idx].lable)
		{
			corr++;
		//	printf("predict %d, act %f\n", maxIdx, mnist->test_data[idx].lable);
		}
		else
		{
		#if 0
			printf("error predict %d, act %f\n", maxIdx, mnist->test_data[idx].lable);
			for(i=0; i<10; i++)
			{
				printf("%f, ", net->o5.cell[i].out);
			}
			printf("\n");
		#endif
		}
	}

	printf("Predict Done(total %d)!(%f)\n", num, corr/(float)num);
	return corr/(float)num;
}

float predict_bacth(LeNet * net,Mnist * mnist)
{
	int i = 0;
	int num = 1000;
	int idx = 0, maxIdx = 0;
	float maxnum = -1, corr = 0;
	MnistImage *image;

	for(idx=0; idx<num; idx++)
	{
		maxnum = -1;
		maxIdx = 0;
		
		image = &mnist->test_data[(idx%100)+(idx/100)*100];
		forward(net, image);
		for(i=0; i<10; i++)
		{
			if(maxnum < net->o5.cell[i].out)
			{
				maxnum = net->o5.cell[i].out;
				maxIdx = i;
			}
		}

		if(maxIdx == image->lable)
		{
			corr++;
		}
		else
		{
	#if 0
			printf("error predict %d, act %f\n", maxIdx, image->lable);
			for(i=0; i<10; i++)
			{
				printf("%f, ", net->o5.cell[i].out);
			}
			printf("\n");
	#endif
		}
	}

	printf("Predict Done(total %d)!(%f)\n", num, corr/(float)num);
	return corr/(float)num;
}


void train_test(LeNet *net, Mnist *mnist)
{
	int idx = 0, i;
	MnistImage *image = &mnist->test_data[0];

	int num05 = 0, num01 = 0, num = 0;
	MnistImage **imageNum05 = malloc(sizeof(MnistImage *)*60000);
	MnistImage **imageNum01 = malloc(sizeof(MnistImage *)*60000);

	for(idx=0; idx<mnist->train_samples; idx++)
	{
		if(mnist->train_data[idx].lable == 5)
		{
			imageNum05[num05] = &mnist->train_data[idx];
			num05++;
		}
		else if(mnist->train_data[idx].lable == 1)
		{
			imageNum01[num01] = &mnist->train_data[idx];
			num01++;
		}
	}

	if(num05 > 1000)
		num05 = 1000;
	if(num01 > 1000)
		num01 = 1000;

	num = num01;
	if(num > num05)
		num = num05;

	for(idx=0; idx<10; idx++)
	{
#if 0	
	if(idx%2 == 0)
		image = imageNum05[idx/2];
	else
		image = imageNum01[idx/2];
#endif
		image = &mnist->test_data[idx];
		matPrint("IN", (float*)image->img, 32, 32);
		matPrint("C1W", (float*)net->c1.cell[0].W, 5, 5);
		printf("C1 Basis:%f\n", net->c1.cell[0].basis);
		stepC1(net, image);
		matPrint("C1OUT", (float*)net->s2.cell[0].c1in, 28, 28);

		stepS2(net);
		matPrint("S2OUT", (float*)net->s2.cell[0].s2out, 14, 14);

		stepC3(net);
		matPrint("C3W", (float*)net->c3.cell[0].W, 5, 5);
		matPrint("C3OUT", (float*)net->s4.cell[0].c3in, 10, 10);

		stepS4(net);
		matPrint("S4OUT", (float*)net->s4.cell[0].s4out, 5, 5);

		stepSpread(net);
		stepO5(net);
		matPrint("O5IN", net->o5.vector, 1, 400);
		matPrint("O5W", net->o5.cell[0].W, 1, 400);
		printf("O5 Basis:%f\n", net->o5.cell[0].basis);
		

		backO5(net, image);
		printf("O5 Delta\n");
		for(i=0; i<10; i++)
			printf("%f ", net->o5.cell[i].delta);
		printf("\n");

		backS4(net);
		matPrint("B-S4", (float*)net->s4.cell[0].dw, 5, 5);

		backC3(net);
		matPrint("B-C3", (float*)net->c3.cell[0].dw, 10, 10);

		

		backS2(net);
		matPrint("B-S2", (float*)net->s2.cell[0].dw, 14, 14);

		backC1(net);
		matPrint("B-C1", (float*)net->c1.cell[0].dw, 28, 28);


		matPrint("C1W", (float*)net->c1.cell[0].W, 5, 5);
		printf("C1 Basis:%f\n", net->c1.cell[0].basis);
		matPrint("C3W", (float*)net->c3.cell[0].W, 5, 5);
		printf("C3 Basis:%f\n", net->c3.cell[0].basis);
		matPrint("O5W", (float*)net->o5.cell[0].W, 1, 400);
		printf("O5 Basis:%f\n", net->o5.cell[0].basis);
		updateWB(net);
		matPrint("D-C1W", (float*)net->c1.cell[0].W, 5, 5);
		printf("C1 Basis:%f\n", net->c1.cell[0].basis);
		matPrint("D-C3W", (float*)net->c3.cell[0].W, 5, 5);
		printf("C3 Basis:%f\n", net->c3.cell[0].basis);
		matPrint("D-O5W", (float*)net->o5.cell[0].W, 1, 400);
		printf("O5 Basis:%f\n", net->o5.cell[0].basis);

		clear(net);
	}

}

void main()
{
	LeNet net, backup;
	Mnist mnist;
	time_t st, end;

	memset(&net, 0, sizeof(net));
	prepareLeNet(&net);
	mnist.test_samples = mnist.train_samples = 0;
	mnist.train_data = mnist.test_data = NULL;

//	loadMnistStatic(&mnist);
	loadMnist(&mnist);
//	saveMnistStatic(&mnist);	//为加快每次读取mnist速度，把mnist的4个文件保存成一个已固化结构的文件
	preprocessMnist(&mnist);
	
	st = time(NULL);
#if 0		//测试神经网络
	loadNet(&net);
	predict(&net, &mnist);
//	train_test(&net, &mnist);

#elif 1		//神经网络训练
//	loadNet(&net);
//	for(int i=0; i<2; i++)
		train(&net, &mnist);
	saveNet(&net);
#else
	float corr, corrmax = 0;
	for(int i=0; i<60; i++)
	{
		train_batch(&net, &mnist, 1000*i, 1000);
		corr = predict_bacth(&net, &mnist);
		if(corr > corrmax)
		{
			corrmax = corr;
			backupNet(&net, &backup);
		}
	}
	saveNet(&backup);
#endif
	end = time(NULL);
	printf("Cost %ld\n", end-st);
}



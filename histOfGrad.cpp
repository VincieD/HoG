

//#include "stdafx.h"
// OpenCV needed for the loading the image and other necessary operation
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "stdafx.h"

using namespace std;
//using namespace cv; // for opencv
#include "histOfGrad.hpp"


///////////////////// Public variables ///////////////////////////
//unsigned char pixelStep;

///////////////////// Private variables  ///////////////////////////

//unsigned char winXsize;
//unsigned char winYsize;

///////////////////// Public functions ///////////////////////////


HistogramOfGradients::HistogramOfGradients()         // set default values of wx, wy and nbin
{

	PI = 180 / 3.14159265;
	// necesary inputs for the HOG calculation
	winXsize = 5;	// HOG descriptor size in axis X
	winYsize = 5;	// HOG descriptor size in axis Y

	inputWindowX = 60;	// calculation of input window X size 
	inputWindowY = 114;	// calculation of input window Y size

	nbinX = unsigned char (inputWindowX / float(winXsize) );		// number of BINs in axis X
	nbinY = unsigned char (inputWindowY / float(winYsize) );		// number of BINs in axis Y
}


// gradients on x direction
void HistogramOfGradients::imfilterGx(vector<vector<unsigned char> >& Im, vector<vector<int> >& grad_xr)
{
	// hx = [-1, 0, 1];
	//grad_xr.clear();
	//grad_xr = Im;
	int TempLeft, TempRight;
	for (int ii = 0; ii<(int)Im.size(); ii++)
	{
		for (int jj = 0; jj<(int)Im[0].size(); jj++)
		{
			TempLeft = (jj - 1<0 ? Im[ii][jj] : Im[ii][jj - 1]);
			TempRight = (jj + 1 >= (int)Im[0].size() ? Im[ii][jj] : Im[ii][jj + 1]);
			grad_xr[ii][jj] = TempRight - TempLeft;
		}
	}
	return;
}

// gradients on y direction
void HistogramOfGradients::imfilterGy(vector<vector<unsigned char> >& Im, vector<vector<int> >& grad_yu)
{
	// hy = [1 0 -1]^T
	//grad_yu.clear();
	//grad_yu = Im;
	int TempUp, TempDown;
	for (int ii = 0; ii<(int)Im.size(); ii++)
	{
		for (int jj = 0; jj<(int)Im[0].size(); jj++)
		{
			TempUp = (ii - 1 < 0 ? Im[ii][jj] : Im[ii - 1][jj]);
			TempDown = (ii + 1 >= (int)Im.size() ? Im[ii][jj] : Im[ii + 1][jj]);
			grad_yu[ii][jj] = TempUp - TempDown;
		}
	}
	return;
}

//
// compute angle and magnitude
void HistogramOfGradients::GetAnglesAndMagnit(vector<vector<int> >& grad_yu, vector<vector<int> >& grad_xr, vector<vector<int> >& angles, vector<vector<int> >& magnit)
{
	for (unsigned int ii = 0; ii<grad_xr.size(); ii++)
	{
		for (unsigned int jj = 0; jj<grad_xr[0].size(); jj++)
		{
			angles[ii][jj] = (int)((atan2((float)(grad_yu[ii][jj]),(float)(grad_xr[ii][jj])))*PI);
		}
	}

	for (unsigned int ii = 0; ii<grad_xr.size(); ii++)
	{
		for (unsigned int jj = 0; jj<grad_xr[0].size(); jj++)
		{
			magnit[ii][jj] = (int)(sqrt(pow((float)(grad_yu[ii][jj]), 2) + pow((float)(grad_xr[ii][jj]), 2)));
		}
	}
	return;
}

//
// compute angle and magnitude
inline int HistogramOfGradients::GetWinMagnit_X( vector<vector<int> >& win_grad_xr)
{
	int temp_X_magnitude = 0;
	
	for (unsigned int ii = 0; ii<win_grad_xr.size(); ii++)
	{
		for (unsigned int jj = 0; jj<win_grad_xr[0].size(); jj++)
		{
			temp_X_magnitude += win_grad_xr[ii][jj];
		}
	}
	return temp_X_magnitude;
}

inline int HistogramOfGradients::GetWinMagnit_Y( vector<vector<int> >& win_grad_yu)
{
	int temp_Y_magnitude = 0;
	
	for (unsigned int ii = 0; ii<win_grad_yu.size(); ii++)
	{
		for (unsigned int jj = 0; jj<win_grad_yu[0].size(); jj++)
		{
			temp_Y_magnitude += win_grad_yu[ii][jj];
		}
	}

	return temp_Y_magnitude;
}

inline int HistogramOfGradients::GetWinAngle( vector<vector<int> >& win_grad_yu,vector<vector<int> >&  win_grad_xr)
{
	int temp_angle = 0;
	
	for (unsigned int ii = 0; ii<win_grad_yu.size(); ii++)
	{
		for (unsigned int jj = 0; jj<win_grad_yu[0].size(); jj++)
		{
			temp_angle += (int)((atan2((float)(win_grad_yu[ii][jj]),(float)(win_grad_xr[ii][jj])))*PI);
		}
	}

	return temp_angle / (win_grad_yu.size() * win_grad_yu[0].size());
}

void HistogramOfGradients::getDescriptor(vector<vector<unsigned char> >& image, vector<vector<unsigned char> >& HogDescriptor)
{
	unsigned char indexX = 0;
	unsigned char indexY = 0;

	long int temp_X_magnitude = 0;
	long int temp_Y_magnitude = 0;
	long int temp_angle = 0;
	unsigned char temp_magnitude = 0;
	bool continueFlag = true;
	int imageSizeX = image[0].size();
	int imageSizeY = image.size();

	// Create local matrix of gradients in x-axis and y-axis
	vector<vector<int> > grad_xr(imageSizeY, vector<int>(imageSizeX));
	vector<vector<int> > grad_yu(imageSizeY, vector<int>(imageSizeX));

	// prevent any array oversizing and program exception
	if (image[0].size() != inputWindowX)
	{
		cout << "Wrong input size of Sliding Window!!!! " << "Aktual window size" << imageSizeX << "px, needed window size: " << inputWindowX << "px. " << endl;
		continueFlag = false;
	}
	if (image.size() != inputWindowY)
	{
		cout << "Wrong input size of Sliding Window!!!! " << "Aktual window size" << imageSizeY << "px, needed window size: " << inputWindowY << "px. " << endl;
		continueFlag = false;
	}

	// calculating the gradients, magnitudes and agles for image sample
	imfilterGx(image, grad_xr);
	imfilterGy(image, grad_yu);

	// -------------------- START - Initial LOOP through the whole Descriptor Window ---------------------------
	if (continueFlag == true)
	{
		for (unsigned int ii = 0; ii <= (imageSizeY - winYsize); ii += nbinX)
		{
			for (unsigned int jj = 0; jj <= (imageSizeX - winXsize); jj += nbinY)
			{
				// -------------------- START - Inner LOOP for bin window calculation ---------------------------
				for (unsigned int i = ii; i < winYsize + ii; i += 1)
				{
					// windows Height shall not be greater than image rows
					if (i>image.size())
					{
						break;
					}

					for (unsigned int j = jj; j < winXsize + jj; j += 1)
					{
						// windows Width shall not be greater than image cols
						if (j>image[0].size())
						{
							break;
						}
						// normalize the vector is necessary - to avoid the picture contrast differences
						temp_X_magnitude += (int)pow(grad_xr[i][j],2);
						temp_Y_magnitude += (int)pow(grad_yu[i][j],2);
					}
				}
				// -------------------- END - Inner LOOP for bin window calculation ---------------------------

				// -------------------- HOGs calculating -------------------- 
				temp_angle = (int)((atan2((float)(sqrt(temp_Y_magnitude)), (float)(sqrt(temp_X_magnitude))))*PI);
				temp_magnitude = (unsigned char)(double(sqrt(temp_X_magnitude)) * cos(temp_angle));

				HogDescriptor[indexY].at(indexX) = temp_magnitude;

				// clearing all variables - ready for new cycle
				temp_X_magnitude = 0;
				temp_Y_magnitude = 0;

				// increasing the X index
				indexX++;
			}
			// increasing the Y index
			indexY++;
			indexX = 0;
		}
		indexY = 0;
	}
	// -------------------- END - Initial LOOP through the whole Descriptor Window ---------------------------
}

unsigned char HistogramOfGradients::getWinXsize()
{
	return inputWindowX;
}

unsigned char HistogramOfGradients::getWinYsize()
{
	return inputWindowY;
}

void HistogramOfGradients::setWinXsize(unsigned char sizeX)
{
	HistogramOfGradients::inputWindowX = sizeX;
	nbinX = unsigned char(inputWindowX / float(winXsize));		// number of BINs in axis X

}

void HistogramOfGradients::setWinYsize(unsigned char sizeY)
{
	HistogramOfGradients::inputWindowY = sizeY;
	nbinY = unsigned char(inputWindowY / float(winYsize));		// number of BINs in axis Y
}

unsigned char HistogramOfGradients::getBinX()
{
	return nbinX;
}

unsigned char HistogramOfGradients::getBinY()
{
	return nbinY;
}

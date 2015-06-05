

//#include "stdafx.h"
// OpenCV needed for the loading the image and other necessary operation
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "stdafx.h"

using namespace std;
//using namespace cv; // for opencv
#include "histOfGrad.hpp"


///////////////////// Public variables ///////////////////////////
unsigned int dim_x;
unsigned int dim_y;
unsigned char pixelStep;

///////////////////// Private variables  ///////////////////////////

unsigned char winXsize;
unsigned char winYsize;

///////////////////// Public functions ///////////////////////////


HistogramOfGradients::HistogramOfGradients()         // set default values of wx, wy and nbin
{
	wx = 5;
	wy = 5;
	nbin = 10;
	PI = 180 / 3.14159265;
	
	pixelStep = 1;
    dim_x = 24;
    dim_y = 32;
    winXsize = 3;
    winYsize = 4;
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
			TempUp = (ii - 1<0 ? Im[ii][jj] : Im[ii - 1][jj]);
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
			temp_angle = (int)((atan2((float)(win_grad_yu[ii][jj]),(float)(win_grad_xr[ii][jj])))*PI);
		}
	}

	return temp_angle;
}

void HistogramOfGradients::getSlidingWindows(vector<vector<unsigned char> >& image, vector<vector<unsigned char> >& HogMatrixMag,vector<vector<int> >& HogMatrixAng)
{
  unsigned char indexX = 0;
  unsigned char indexY = 0;
  unsigned char winWidth = getWinXsize();
  unsigned char winHeight = getWinYsize();
  
  int temp_X_magnitude = 0;
  int temp_Y_magnitude = 0;
  int temp_angle = 0;
  int temp_magnitude = 0;
  
  // Create local matrix of gradients in x-axis and y-axis
  vector<vector<int> > grad_xr(dim_y, vector<int>(dim_x));
  vector<vector<int> > grad_yu(dim_y, vector<int>(dim_x));
  
  // Create local matrix of magnitudes and agles
  vector<vector<int> > magnitude(dim_y, vector<int>(dim_x));
  vector<vector<int> > angles(dim_y, vector<int>(dim_x));
  
  // Create local matrix for sliding window
  vector<vector<unsigned char> > window(winHeight, vector<unsigned char>(winWidth));
  vector<vector<int> > win_grad_xr(winHeight, vector<int>(winWidth));
  vector<vector<int> > win_grad_yu(winHeight, vector<int>(winWidth));
  
  // calculating the gradients, magnitudes and agles for image sample
  imfilterGx(image, grad_xr);
  imfilterGy(image, grad_yu);
  GetAnglesAndMagnit(grad_yu, grad_xr, angles, magnitude);
  
  for (unsigned int ii = 0; ii<HogMatrixMag.size(); ii++)
  	{
		for (unsigned int jj = 0; jj<HogMatrixMag[0].size(); jj++)
      {
  
		for (unsigned int i = ii; i < winHeight + ii; i += 1)
      	{
      		// windows Height shall not be greater than image rows
			if (i>image.size())
      			{break;}
      	  
			for (unsigned int j = jj; j < winWidth + jj; j += 1)
      		{
      			// windows Width shall not be greater than image cols
				if (j>image[0].size())
      				{break;}
      				
      			// index of our windows shall be inside the limits	
      			if((indexX > winWidth) || (indexY > winHeight))
      				{break;}		
      			
      			// append to the rects another vector which is called rect
               window[indexY].at(indexX) = (unsigned char)(image[i][j]);
               win_grad_xr[indexY].at(indexX) = (int)(grad_xr[i][j]);
               win_grad_yu[indexY].at(indexX) = (int)(grad_yu[i][j]);
               //cout << "index X: " << (int)indexX << " index Y: " << (int)indexY << endl;
               indexX++;
      		}
      		indexY++;
      	    
             // clearing the index otherwise it will overflow
      		if(indexX > winWidth - 1)
      		{indexX = 0;}
      		// clearing the index otherwise it will overflow
      		if(indexY > winHeight - 1)
      		{indexY = 0;}
      	}
	      //indexY = 0;
	      //indexX = 0;
	      
        // START HOGs calculating
          temp_X_magnitude = (GetWinMagnit_X(win_grad_xr))/(winWidth*winHeight);
          temp_Y_magnitude = (GetWinMagnit_Y(win_grad_yu))/(winWidth*winHeight);
          temp_angle = (int)((atan2((float)(temp_Y_magnitude),(float)(temp_X_magnitude)))*PI);
          temp_magnitude = (unsigned char) (temp_X_magnitude * cos(temp_angle));
                
          HogMatrixMag[ii].at(jj) = temp_magnitude;
          HogMatrixAng[ii].at(jj) = temp_angle;
       }
   }
       // END HOGs calculating
    
}



unsigned char HistogramOfGradients::getWinXsize()
{
   return winXsize;         
}

unsigned char HistogramOfGradients::getWinYsize()
{
   return winYsize;         
}

void HistogramOfGradients::setWinXsize(unsigned char sizeX)
{
   HistogramOfGradients::winXsize = sizeX;         
}

void HistogramOfGradients::setWinYsize(unsigned char sizeY)
{
   HistogramOfGradients::winYsize = sizeY;         
}

unsigned int HistogramOfGradients::getPicXsize()
{
	return dim_x;
}

unsigned int HistogramOfGradients::getPicYsize()
{
	return dim_y;
}

void HistogramOfGradients::setPicXsize(unsigned int sizeX)
{
	HistogramOfGradients::dim_x = sizeX;
}

void HistogramOfGradients::setPicYsize(unsigned int sizeY)
{
	HistogramOfGradients::dim_y = sizeY;
}


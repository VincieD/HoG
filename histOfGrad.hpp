// Usage: the main function for HoG class is HOGdescriptor, with two inputs:
//        Im - The input 2-dimensional vector, usually from grayscale image
//        descriptor - The output HoG feature with wx*wy*nbin dimensions
// The user can change the values of wx, wy and nbin, which are x-window size, y-window size, and angle bins number

#ifndef __HISTOFGRADIENT_H_INCLUDED__
#define __HISTOFGRADIENT_H_INCLUDED__

   #include <iostream>
   #include <fstream>
   #include <iostream>
   #include <vector>
   #include <map>
   #include <string>
   #include <algorithm>
   #include <cmath>
   #include <stdlib.h>
   #include <stdio.h>
   #include <math.h>
   #include <time.h>


   class HistogramOfGradients
   {
   	public:
   		HistogramOfGradients();
   		int wx;
   		int wy;
   		int nbin;
         unsigned char pixelStep;
         
   		void getSlidingWindows(std::vector<std::vector<unsigned char> >& image, std::vector<std::vector<unsigned char> >& HogMatrixMag,std::vector<std::vector<int> >& HogMatrixAng);
   		void imfilterGx(std::vector<std::vector<unsigned char> >& Im, std::vector<std::vector<int> >& grad_xr);
   		void imfilterGy(std::vector<std::vector<unsigned char> >& Im, std::vector<std::vector<int> >& grad_yu);
   		void GetAnglesAndMagnit(std::vector<std::vector<int> >& grad_yu, std::vector<std::vector<int> >& grad_xr, std::vector<std::vector<int> >& angles, std::vector<std::vector<int> >& magnit);
//   		void GetVector2Range(vector<vector<float>>& inVec, int L1, int L2, int C1, int C2,vector<vector<float>>& outVec);
//   		void StraitenVector(vector<vector<float>>& inVec, vector<float>& outLine);
//   		float L2NormVec1(vector<float>& inVec);

         unsigned char getWinXsize();
         unsigned char getWinYsize();
         void setWinXsize(unsigned char sizeX);
         void setWinYsize(unsigned char sizeY);

		 unsigned int getPicXsize();
		 unsigned int getPicYsize();
		 void setPicXsize(unsigned int sizeX);
		 void setPicYsize(unsigned int sizeY);
   
   	private:
   		double PI;
         unsigned char winXsize;
         unsigned char winYsize;

		 unsigned int dim_x;
		 unsigned int dim_y;
         
         inline int GetWinMagnit_X( std::vector<std::vector<int> >& win_grad_xr);
         inline int GetWinMagnit_Y( std::vector<std::vector<int> >& win_grad_yu);
         inline int GetWinAngle( std::vector<std::vector<int> >& win_grad_yu, std::vector<std::vector<int> >&  win_grad_xr);
         
   };

#endif 

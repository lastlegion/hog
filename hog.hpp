#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

//Parameters
#define N_BINS 16           //Number of bins
#define N_DIVS 3            //Number of cells = N_DIVS*N_DIVS
#define N_PHOG N_DIVS*N_DIVS*N_BINS
#define BIN_RANGE (2*CV_PI)/N_BINS
//Haar Cascade Path
string face_cascade_path = "haarcascade_frontalface_default.xml";


//Input: Grayscale image
//Output: HOG features
Mat hog(const Mat &Img)
{
    Mat Hog;
    Hog = Mat::zeros(1, N_PHOG, CV_32FC1);

    Mat Ix, Iy;

    //Find orientation gradients in x and y directions
    Sobel(Img, Ix, CV_16S, 1, 0, 3);
    Sobel(Img, Iy, CV_16S, 0, 1, 3);
    
    int cellx = Img.cols/N_DIVS;
    int celly = Img.rows/N_DIVS;
    
    int img_area = Img.rows * Img.cols;

    for(int m=0; m < N_DIVS; m++)
    {
        for(int n=0; n < N_DIVS; n++)
        {
             for(int i=0; i<cellx; i++)
             {
                for(int j=0; j<celly; j++)
                {
                    
                    float px, py, grad, norm_grad, angle, nth_bin;

                    //px = Ix.at(m*cellx+i, n*celly+j);
                    //py = Iy.at(m*cellx+i, n*celly+j);
                    px = static_cast<float>(Ix.at<int16_t>((m*cellx)+i, (n*celly)+j ));
                    py = static_cast<float>(Iy.at<int16_t>((m*cellx)+i, (n*celly)+j ));

                    grad = static_cast<float>(std::sqrt(1.0*px*px + py*py));
                    norm_grad = grad/img_area;

                    //Orientation
                    angle = std::atan2(py,px);
                    
                    //convert to 0 to 360 (0 to 2*pi)
                    if( angle < 0)
                        angle+= 2*CV_PI;
                    
                    //find appropriate bin for angle
                    nth_bin = angle/BIN_RANGE;
                    
                    //add magnitude of the edges in the hog matrix
                    Hog.at<float>(0,(m*N_DIVS +n)*N_BINS + static_cast<int>(angle)) += norm_grad;

                }
             }
        }
    }  

    //Normalization
    for(int i=0; i< N_DIVS*N_DIVS; i++)
    { 
        float max=0;
        int j;
        for(j=0; j<N_BINS; j++)
        {
            if(Hog.at<float>(0, i*N_BINS+j) > max)
                max = Hog.at<float>(0,i*N_BINS+j);
        }
        for(j=0; j<N_BINS; j++)
            Hog.at<float>(0, i*N_BINS+j)/=max;
    }
    return Hog;
}



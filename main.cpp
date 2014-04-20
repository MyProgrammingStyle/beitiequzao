#include<opencv2/opencv.hpp> 
#include<iostream>
#include<vector>
#include<map>
#include<fstream>
#include<cmath>
#define SQRT2 1.4142135623730950488016887242097
#define PI 3.1415926535897932384626433832795
using namespace cv;
using namespace std;
namespace core_denoising
{
  inline int cul_youchang_0(const vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((col - 1 - i >= 0) && (source[col - i - 1][row] != 0))
      i++;
    while ((col + 1 + j < source.size()) && (source[col + j + 1][row] != 0))
      j++;
    return i + j + 1;
  }
  inline int cul_youchang_45(const vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row + 1 + i < source[0].size()) && (col - 1 - i >= 0) && (source[col - i - 1][row + i + 1] !=0))
      i++;
    while ((col + 1 + j < source.size()) && (row - 1 - j >= 0 && (source[col + j + 1][row - j - 1] !=0)))
      j++;
    return (i + j + 1) * SQRT2;
  }
  inline int cul_youchang_90(const vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row - 1 - i >= 0) && (source[col][row - i - 1] != 0))
      i++;
    while ((row + 1 + j < source[0].size()) && (source[col][row + j + 1] != 0))
      j++;
    return i + j + 1;
  }
  inline int cul_youchang_135(const vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row - 1 - i >= 0) && (col - 1 - i >= 0) && (source[col - i - 1][row - i - 1] != 0))
      i++;
    while ((col + 1 + j < source.size()) && (row + 1 + j < source[0].size()) && (source[col + i + 1][row + j + 1] != 0))
      j++;
    return (i + j + 1) * SQRT2;
  }
  int youchang_vertical(const vector<vector<short>>& source,int col,int row)
  {
    int r=cul_youchang_90(source,col,row);
    int l;
    for(int i=1;i<4;i++)
      {
	l=cul_youchang_0(source,col+i*0.25,row);
	if(l<(r/3.0))
	  return 0;
      }
    return r;
  }
  int youchang_horizontal(const vector<vector<short>>& source,int col,int row)
  {
    int r=cul_youchang_0(source,col,row);
    int l;
    for(int i=1;i<4;i++)
      {
	l=cul_youchang_90(source,col,row+i*0.25);
	if(l<(r/3.0))
	  return 0;
      }
    return r;
  }
  //////////////////////////////
  vector<int> youchang(const vector<vector<short>>& source)//, vector<vector<short>>*  result)
  {
    const int rows = source.size(), columns = source[0].size();
    for (auto tem : source)
      {
	if (tem.size() != columns)
	  throw(1);
      }
    int max_length;
    if (rows > columns)
      max_length = rows;
    else
      max_length = columns;
    vector<int> rs(max_length+1);
    for(int i=1;i<rows-1;i++)
      for(int j=1;j<columns-1;j++)
	{
	  if (source[i][j] != 0 && source[i + 1][j] != 0 && source[i - 1][j] == 0)
	    rs[youchang_horizontal(source, i, j)]++;
	  if (source[i][j] != 0 && source[i][j + 1] != 0 && source[i][j - 1] == 0)
	    rs[youchang_vertical(source, i, j)]++;
	}
    return rs;
  }
}
double** newDoubleMatrix(int nx, int ny)
{
  double** matrix = new double*[ny];

  for (int i = 0; i < ny; i++)
    {
      matrix[i] = new double[nx];
    }
  if (!matrix)
    return nullptr;
  return
    matrix;
}
bool deleteDoubleMatrix(double** matrix, int nx, int ny)
{
  if (!matrix)
    {
      return true;
    }
  for (int i = 0; i < ny; i++)
    {
      if (matrix[i])
	{
	  delete[] matrix[i];
	}
    }
  delete[] matrix;

  return true;
}
Mat tv_denoising(Mat img, int iter=50)
{
  int ep = 1;
  int nx = img.cols;
  int ny = img.rows;
  double dt = 0.25f;
  double lam = 0.01;
  int ep2 = ep*ep;

  double** image = newDoubleMatrix(nx, ny);
  double** image0 = newDoubleMatrix(nx, ny);

  for (int i = 0; i<ny; i++){
    uchar* p = img.ptr<uchar>(i);
    for (int j = 0; j<nx; j++){
      image0[i][j] = image[i][j] = (double)p[j];
    }
  }
  for (int t = 1; t <= iter; t++){ 
    for (int i = 0; i < ny; i++){
      for (int j = 0; j < nx; j++){
	int tmp_i1 = (i + 1)<ny ? (i + 1) : (ny - 1);
	int tmp_j1 = (j + 1)<nx ? (j + 1) : (nx - 1);
	int tmp_i2 = (i - 1) > -1 ? (i - 1) : 0;
	int tmp_j2 = (j - 1) > -1 ? (j - 1) : 0;
	double tmp_x = (image[i][tmp_j1] - image[i][tmp_j2]) / 2; //I_x  = (I(:,[2:nx nx])-I(:,[1 1:nx-1]))/2;  
	double tmp_y = (image[tmp_i1][j] - image[tmp_i2][j]) / 2; //I_y  = (I([2:ny ny],:)-I([1 1:ny-1],:))/2;  
	double tmp_xx = image[i][tmp_j1] + image[i][tmp_j2] - image[i][j] * 2; //I_xx = I(:,[2:nx nx])+I(:,[1 1:nx-1])-2*I;  
	double tmp_yy = image[tmp_i1][j] + image[tmp_i2][j] - image[i][j] * 2; //I_yy = I([2:ny ny],:)+I([1 1:ny-1],:)-2*I;  
	double tmp_dp = image[tmp_i1][tmp_j1] + image[tmp_i2][tmp_j2]; //Dp   = I([2:ny ny],[2:nx nx])+I([1 1:ny-1],[1 1:nx-1]);  
	double tmp_dm = image[tmp_i2][tmp_j1] + image[tmp_i1][tmp_j2]; //Dm   = I([1 1:ny-1],[2:nx nx])+I([2:ny ny],[1 1:nx-1]);  
	double tmp_xy = (tmp_dp - tmp_dm) / 4; //I_xy = (Dp-Dm)/4;  
	double tmp_num = tmp_xx*(tmp_y*tmp_y + ep2)
	  - 2 * tmp_x*tmp_y*tmp_xy + tmp_yy*(tmp_x*tmp_x + ep2); //Num = I_xx.*(ep2+I_y.^2)-2*I_x.*I_y.*I_xy+I_yy.*(ep2+I_x.^2);  
	double tmp_den = pow((tmp_x*tmp_x + tmp_y*tmp_y + ep2), 1.5); //Den = (ep2+I_x.^2+I_y.^2).^(3/2);  
	image[i][j] += dt*(tmp_num / tmp_den + lam*(image0[i][j] - image[i][j]));
      }
    }


  }

  Mat new_img;
  img.copyTo(new_img);
  for (int i = 0; i<img.rows; i++){
    uchar* p = img.ptr<uchar>(i);
    uchar* np = new_img.ptr<uchar>(i);
    for (int j = 0; j<img.cols; j++){
      int tmp = (int)image[i][j];
      tmp = max(0, min(tmp, 255));
      np[j] = (uchar)(tmp);
    }
  }
 
  deleteDoubleMatrix(image0, nx, ny);
  deleteDoubleMatrix(image, nx, ny);
  return new_img;
}
int main()
{
  Mat im_gray = imread("C:\\Users\\明鸣\\Pictures\\邓平.png", CV_LOAD_IMAGE_GRAYSCALE);
  //IplImage* src;
  //IplImage* src1;
  //src = cvLoadImage("C:\\Users\\Public\\Pictures\\Sample Pictures\\Tulips.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  Mat im_gray2 = tv_denoising(im_gray,1);
  Mat im_gray3;
  int i,j;
  ofstream eng;
  eng.open("c.txt");
  threshold(im_gray2, im_gray3, 128, 255, CV_THRESH_OTSU);
  cvNamedWindow("lena1", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("lena2", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("lena3", CV_WINDOW_AUTOSIZE);
  //cvShowImage("lena", src);
  imshow("lena3", im_gray);
  imshow("lena1", im_gray2);
  vector<vector<short>> rs(im_gray3.rows);
  for (auto& i : rs)
    i.resize(im_gray3.cols);
  for (i = 0; i < im_gray3.rows; i++)
    {
      for (j = 0; j < im_gray3.cols; j++)
	rs[i][j] = (int)((bool)im_gray3.ptr(i)[j]);
      cout << "\n";
    } 	
  for (i = 0; i < im_gray3.rows; i++)
    {
      for (j = 0; j < im_gray3.cols; j++)
	//eng << rs[i][j];
	cout << "\n";
    }
  vector<int> ansss=core_denoising::youchang(rs);
  for (auto i : ansss)eng << " " << i;
  //eng << im_gray3.rows << endl << im_gray3.cols;
  //im_gray3.ptr(im_gray3.rows-1)[im_gray3.cols-1];
  // for (j = 0; j < ;j++)
  // im_gray3.ptr(i)[j]=255;
  imshow("lena2", im_gray3);
  cvWaitKey(0);
  cvDestroyWindow("lena");
  //cvReleaseImage(&src);
  return 0;
}
//两混合正态分布的参数估计方法 基于缺失数据的两正态混合分布的参数估计


//《黄冈师范学院学报》2006年 第3期 | 杨珂玲 韩慧芳

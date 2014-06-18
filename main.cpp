#include<opencv2/opencv.hpp> 
#include<iostream>
#include<vector>
#include<list>
#include<fstream>
#include<cmath>
#include<stack>
//#define DEBUGING
//#define HOUGH
#define SQRT2 1.4142135623730950488016887242097
using namespace cv;
using namespace std;

namespace core_denoising
{
  inline float cul_youchang_0(const vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((col - 1 - i >= 0) && (source[col - i - 1][row] != 0))
      i++;
    while ((col + 1 + j < source.size()) && (source[col + j + 1][row] != 0))
      j++;
    return i + j + 1;
  }
  inline float cul_youchang_45(const vector<vector<short>>& source, int col, int row)

  {
    int i = 0, j = 0;
    while ((row + 1 + i < source[0].size()) && (col - 1 - i >= 0) && (source[col - i - 1][row + i + 1] != 0))
      i++;
    while ((col + 1 + j < source.size()) && (row - 1 - j >= 0 && (source[col + j + 1][row - j - 1] != 0)))
      j++;
    return (i + j + 1) * SQRT2;
  }
  inline float cul_youchang_90(const vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row - 1 - i >= 0) && (source[col][row - i - 1] != 0))
      i++;
    while ((row + 1 + j < source[0].size()) && (source[col][row + j + 1] != 0))
      j++;
    return i + j + 1;
  }
  inline float cul_youchang_135(const vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row - 1 - i >= 0) && (col - 1 - i >= 0) && (source[col - i - 1][row - i - 1] != 0))
      i++;
    while ((col + 1 + j < source.size()) && (row + 1 + j < source[0].size()) && (source[col + j + 1][row + j + 1] != 0))
      j++;
    return (i + j + 1) * SQRT2;
  }
  int youchang_vertical(const vector<vector<short>>& source, int col, int row)
  {
    int r = cul_youchang_90(source, col, row);
    int l;
    for (int i = 1; i<4; i++)
      {
	l = cul_youchang_0(source, col + i*0.25, row);
	if (l<(r / 3.0))
	  return 0;
      }
    return r;
  }
  int youchang_horizontal(const vector<vector<short>>& source, int col, int row)
  {
    int r = cul_youchang_0(source, col, row);
    int l;
    for (int i = 1; i<4; i++)
      {
	l = cul_youchang_90(source, col, row + i*0.25);
	if (l<(r / 3.0))
	  return 0;
      }
    return r;
  }
  //////////////////////////////
  inline int fill_0(vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((col - 1 - i >= 0) && (source[col - i - 1][row] != 0))
      {
	source[col - i - 1][row] = -1;
	i++;
      }

    while ((col + 1 + j < source.size()) && (source[col + j + 1][row] != 0))
      {
	source[col + j + 1][row] = -1;
	j++;
      }
    return i + j + 1;
  }
  inline int fill_45(vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row + 1 + i < source[0].size()) && (col - 1 - i >= 0) && (source[col - i - 1][row + i + 1] != 0))
      {
	source[col - i - 1][row + i + 1] = -1;
	i++;
      }
    while ((col + 1 + j < source.size()) && (row - 1 - j >= 0 && (source[col + j + 1][row - j - 1] != 0)))
      {
	source[col + j + 1][row - j - 1] = -1;
	j++;
      }
    return (i + j + 1) * SQRT2;
  }
  inline int fill_90(vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row - 1 - i >= 0) && (source[col][row - i - 1] != 0))
      {
	source[col][row - i - 1] = -1;
	i++;
      }
    while ((row + 1 + j < source[0].size()) && (source[col][row + j + 1] != 0))
      {
	source[col][row + j + 1] = -1;
	j++;
      }
    return i + j + 1;
  }
  inline int fill_135(vector<vector<short>>& source, int col, int row)
  {
    int i = 0, j = 0;
    while ((row - 1 - i >= 0) && (col - 1 - i >= 0) && (source[col - i - 1][row - i - 1] != 0))
      {
	source[col - i - 1][row - i - 1] = -1;
	i++;
      }
    while ((col + 1 + j < source.size()) && (row + 1 + j < source[0].size()) && (source[col + j + 1][row + j + 1] != 0))
      {
	source[col + j + 1][row + j + 1] = -1;
	j++;
      }
    return (i + j + 1) * SQRT2;
  }
  vector<vector<short>> fill(vector<vector<short>> source, int ans)
  {
    const int rows = source.size(), columns = source[0].size();
    for (auto tem : source)
      {
	if (tem.size() != columns)
	  throw(1);
      }
    for (int i = 1; i<rows - 1; i++)
      for (int j = 1; j<columns - 1; j++)
	{
	  if (source[i][j] == 1)
	    {
	      if (cul_youchang_90(source, i, j) >= ans)
		{
		  fill_90(source, i, j);
		  continue;
		}
	      if (cul_youchang_0(source, i, j) >= ans)
		{
		  fill_0(source, i, j);
		  continue;
		}
	      if (cul_youchang_45(source, i, j) >= ans)
		{
		  fill_45(source, i, j);
		  continue;
		}
	      if (cul_youchang_135(source, i, j) >= ans)
		{
		  fill_135(source, i, j);
		  continue;
		}
	    }
	}
    for (auto& i : source)
      for (auto& j : i)
	{
	  if (j == -1)
	    j = 1;
	  else j = 0;
	}
    return source;
  }
  vector<int> youchang(const vector<vector<short>>& source)//, vector<vector<short>>*  result)
  {
    const int rows = source.size(), columns = source[0].size();
    for (auto tem : source)
      {
	if (tem.size() != columns)
	  throw(1);
      }
    int g;
    int max_length;
    if (rows > columns)
      max_length = rows;
    else
      max_length = columns;
    vector<int> rs;
    for (int i = 1; i<rows - 1; i++)
      for (int j = 1; j<columns - 1; j++)
	{
	  if (source[i][j] != 0 && source[i + 1][j] != 0 && source[i - 1][j] == 0)
	    {
	      g = youchang_horizontal(source, i, j);
	      if (g)
		rs.push_back(g);
	    }
	  if (source[i][j] != 0 && source[i][j + 1] != 0 && source[i][j - 1] == 0)
	    {
	      g = youchang_vertical(source, i, j);
	      if (g)
		rs.push_back(g);
	    }
	}
    return rs;
  }
int p_fill(int x, int y, vector<vector<short>>& a, vector<vector<short>>& b, int ans, int& size, int& sizep)
{
  stack<int,list<int>> xx;
  stack<int,list<int>> yy;
  xx.push(x);
  yy.push(y);
  while(!xx.empty())
    {
      x=xx.top();
      xx.pop();
      y=yy.top();
      yy.pop();
      b[x][y]=0;
      size++;
      if (cul_youchang_90(a, x, y) >= ans)
	{
	  sizep++;
	}
      else if (cul_youchang_45(a, x, y) >= ans)
	{
	  sizep++;
	}
      else if (cul_youchang_135(a, x, y) >= ans)
	{
	  sizep++;
	}
      else if (cul_youchang_0(a, x, y) >= ans)
	{
	  sizep++;
	}
      if (!((x-1<0) || (y<0) || (x-1 >= a.size()) || (y >= a[0].size()) || (b[x-1][y] == 0)))
	{
	  xx.push(x-1);
	  yy.push(y);
	}
      if (!((x+1<0) || (y<0) || (x+1 >= a.size()) || (y >= a[0].size()) || (b[x+1][y] == 0)))
	{
	  xx.push(x+1);
	  yy.push(y);
	}
      if (!((x<0) || (y-1<0) || (x >= a.size()) || (y-1 >= a[0].size()) || (b[x][y-1] == 0)))
	{
	  xx.push(x);
	  yy.push(y-1);
	}
      if (!((x<0) || (y+1<0) || (x >= a.size()) || (y+1 >= a[0].size()) || (b[x][y+1] == 0)))
	{
	  xx.push(x);
	  yy.push(y+1);
	}
    }
}

  int pure_fill(int x, int y, vector<vector<short>>& a)
  {
    if ((x<0) || (y<0) || (x >= a.size()) || (y >= a[0].size()) || (a[x][y] == 0))
      return 1;
    a[x][y] = 0;
    pure_fill(x - 1, y, a);
    pure_fill(x + 1, y, a);
    pure_fill(x, y - 1, a);
    pure_fill(x, y + 1, a);
    return 0;
  }
  vector<vector<short>>
  search_and_fill(vector<vector<short>> a, float p, int ans)
  {
    int size, sizep;
    float k = 0;
    vector<vector<short>> b = a;
    for (int i = 0; i<a.size(); i++)
      for (int j = 0; j<a[0].size(); j++)
	{
	  size = 0, sizep = 0;
	  if ((a[i][j] == 1) && (b[i][j] == 1))
	    {
	      p_fill(i, j, a, b, ans,size,sizep);
	      if ((float)sizep/size<p)
		pure_fill(i, j, a);
	    }
	}
    return a;
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
Mat tv_denoising(Mat img, int iter = 50)
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
	double tmp_x = (image[i][tmp_j1] - image[i][tmp_j2]) / 2;
	double tmp_y = (image[tmp_i1][j] - image[tmp_i2][j]) / 2;
	double tmp_xx = image[i][tmp_j1] + image[i][tmp_j2] - image[i][j] * 2;
	double tmp_yy = image[tmp_i1][j] + image[tmp_i2][j] - image[i][j] * 2;
	double tmp_dp = image[tmp_i1][tmp_j1] + image[tmp_i2][tmp_j2];
	double tmp_dm = image[tmp_i2][tmp_j1] + image[tmp_i1][tmp_j2];
	double tmp_xy = (tmp_dp - tmp_dm) / 4;
	double tmp_num = tmp_xx*(tmp_y*tmp_y + ep2)
	  - 2 * tmp_x*tmp_y*tmp_xy + tmp_yy*(tmp_x*tmp_x + ep2);
	double tmp_den = pow((tmp_x*tmp_x + tmp_y*tmp_y + ep2), 1.5);
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
int main(int argc, char** argv)
{
  Mat im_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  cout << "TV denoising..." << endl;
#ifdef DEBUGING
  Mat im_gray2 = tv_denoising(im_gray, 1);
#else
  Mat im_gray2 = tv_denoising(im_gray);
#endif
  Mat im_gray3;
  int i, j;
  cout << "Thresholding..." << endl;
  threshold(im_gray2, im_gray3, 128, 255, CV_THRESH_OTSU);
  //  cvNamedWindow("lena1", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("lena2", CV_WINDOW_AUTOSIZE);
  //  cvNamedWindow("lena3", CV_WINDOW_AUTOSIZE);
  //cvShowImage("lena", src);
  // imshow("lena3", im_gray);
  //imshow("lena1", im_gray2);
  cout << "Count..." << endl;
  vector<vector<short>> rs(im_gray3.rows);
  for (auto& i : rs)
    i.resize(im_gray3.cols);
  for (i = 0; i < im_gray3.rows; i++)
    {
      for (j = 0; j < im_gray3.cols; j++)
	rs[i][j] = (int)((bool)im_gray3.ptr(i)[j]);
    }
  vector<int> ansss = core_denoising::youchang(rs);
  ofstream eng;
  eng.open("cc.txt");
  cout<<"33333";
  int i3,j3;
  cin>>i3>>j3;
  for(auto sssss:ansss)
    {if((sssss>=i3)&&(sssss<=j3)){ eng<<sssss<<" ";}}
  int nsamples = ansss.size();
  int ncluster = 2;
  Mat samples(ansss);
  cout << "Analysing... " << endl;
  EM em_model = EM(ncluster, EM::COV_MAT_SPHERICAL);

  if (!em_model.train(samples)) {
    cerr << "error training the EM model" << endl;
    exit(-1);
  }

  const Mat& means = em_model.get<Mat>("means");
  double mean1 = means.at<double>(0, 0);
  double mean2 = means.at<double>(1, 0);
  cout << "mean1 = " << mean1 << ", mean2 = " << mean2 << endl;

  const vector<Mat>& covs = em_model.get<vector<Mat> >("covs");
  double scale1 = covs[0].at<double>(0, 0);
  double scale2 = covs[1].at<double>(0, 0);
  cout << "scale1 = " << scale1 << ", scale2 = " << scale2 << endl;
  if (mean1<mean2)
    mean1 = mean2;
  int deal;
  cout << "pls ins deal:";
  cin >> deal;
  float p = 0.8;
  cout << "pls ins p:";
  cin >> p;
  rs = core_denoising::search_and_fill(rs, p, mean1 + deal);
  for (i = 1; i < im_gray3.rows - 1; i++)
    {
      for (j = 1; j < im_gray3.cols - 1; j++)
	{
	  if (rs[i - 1][j] == rs[i + 1][j])
	    rs[i][j] = rs[i - 1][j];
	  if (rs[i][j - 1] == rs[i][j + 1])
	    rs[i][j] = rs[i][j - 1];
	  im_gray3.ptr(i)[j] = 255 * rs[i][j];
	}
    }



  vector<Vec4i> lines;
#ifdef HOUGH
  HoughLinesP(im_gray3, lines, 1, CV_PI / 180, 1000, 3, 0);
  for (size_t i = 0; i < lines.size(); i++)
    {
      Vec4i l = lines[i];
      line(im_gray3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, CV_AA);
    }
  HoughLinesP(im_gray3, lines, 1, CV_PI / 45, 1000, 3, 0);
  for (size_t i = 0; i < lines.size(); i++)
    {
      Vec4i l = lines[i];
      line(im_gray3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, CV_AA);
    }
  HoughLinesP(im_gray3, lines, 1, CV_PI / 90, 1000, 3, 0);
  for (size_t i = 0; i < lines.size(); i++)
    {
      Vec4i l = lines[i];
      line(im_gray3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, CV_AA);
    }
  HoughLinesP(im_gray3, lines, 1, CV_PI / 135, 1000, 3, 0);
  for (size_t i = 0; i < lines.size(); i++)
    {
      Vec4i l = lines[i];
      line(im_gray3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, CV_AA);
    }
#endif
  imshow("lena2", im_gray3);
  imwrite(argv[2], im_gray3);
  cvWaitKey(0);
  cvDestroyWindow("lena");
}

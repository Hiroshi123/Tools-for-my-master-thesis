#include <iostream>
#include <vector>
//#include "lib/bnnSub.cpp"

using namespace std;

template<typename T>
vector<vector<T>> transpose(vector<vector<T>>v){
  
  vector<vector<T>> o(v[0].size(),vector<T>(v.size(),0));
  
  for(int i = 0; i<v.size();i++)
    for(int j = 0; j<v[0].size();j++)
      o[j][i] = v[i][j]; 
  
  return o;
};


vector<vector<float>> dot(vector<vector<float>> x,vector<vector<float>> y){
  
  int x_r = x.size();
  int x_c = x[0].size();
  int y_r = y.size();
  int y_c = y[0].size();
  
  if(x_c != y_r){
    cout << x_c << " :: " <<  y_r << endl;
    cerr << "1st column and 2nd row must be identical" << endl;
  }
  
  vector<vector<float>> z;
  z = vector<vector<float>> (x_r,vector<float>(y_c,0));
  
  //cout << x_r << y_c << endl;
  //iterator based multlix multiplication should be faster but
  // not properly working..
  
  /*
  vector<vector<float> >::iterator xRow = x.begin();
  vector<float>::iterator xCol = xRow->begin();
  vector<vector<float> >::iterator yRow = y.begin();
  vector<float>::iterator yCol = yRow->begin();
  vector<vector<float> >::iterator zRow = z.begin();
  vector<float>::iterator zCol = zRow->begin();
  
  while( xRow != x.end() && yRow != y.end() ){
    while( xCol != xRow->end() && zRow != z.end() ){
      while( yCol != yRow->end() && zCol != zRow->end()){
	*zCol += (*xCol) * (*yCol) ;
	cout << *zCol << endl;
	yCol ++;
	zCol ++;
      }
      xRow ++;
      zRow ++;
    }
    xCol ++;
    yRow ++;
  }
  */
  
  for(int i = 0 ; i < x_c ; i++)
    for(int j = 0 ; j < x_r ; j++)
      for(int k = 0 ; k < y_c ; k++)
	z[j][k] += x[j][i] * y[i][k];
  
  return z;
    
};


//this is map operation


vector<vector<float>> ite(float (*func)(float),vector<vector<float>>v){
  
  vector<vector<float> >::iterator row;
  vector<float>::iterator col;
  for (row = v.begin(); row != v.end(); ++row) {
    for (col = row->begin(); col != row->end(); ++col) {
      *col = func(*col);
    }
  }
  
  return v;
  
}

vector<vector<vector<vector<float>>>> ite4(float (*func)(float),vector<vector<vector<vector<float>>>>v){
  
  vector<vector<vector<vector<float>>>> :: iterator i;
  vector<vector<vector<float>>> :: iterator j;
  vector<vector<float> >::iterator k;
  vector<float>::iterator l;
  
  for (i = v.begin(); i != v.end(); ++i) {
    for (j = i->begin(); j != i->end(); ++j) {
      for (k = j->begin(); k != j->end(); ++k) {
	for (l = k->begin(); l != k->end(); ++l) {
	  *l = func(*l);
	}
      }
    }
  }
  
  return v;
  
}


vector<vector<float>> convolution(vector<vector<float>> data,vector<vector<float>> kernel){
  
  //cout << "are you here ? " << endl;
  
  int dataW   = data.size();
  int dataH   = data[0].size();
  int kernelW = kernel.size();
  int kernelH = kernel[0].size();
  int row,column;
  
  vector<vector<float>> output(dataW,vector<float>(dataH,0));
  
  // simplest implementaion of convolution should be 4 loop.
  
  for(int dw = 0 ;  dw < dataW ; dw++ )
    for(int dh = 0 ;  dh < dataH ; dh++ )
      for(int kw = 0 ; kw < kernelW ; kw ++)
	for(int kh = 0 ; kh < kernelH ; kh++){
	  row    = dw + kw - (kernelW / 2);
	  column = dh + kh - (kernelH / 2);
	  //cout << row << " ; " << column << endl;
	  //cout << dataW << endl;
	  if( row    >= 0 && row    < dataW-1 &&
	      column >= 0 && column < dataH-1 ){
	    //cout << "oep" << endl;
	    output[dw][dh] += data[row][column] * kernel[kw][kh];
	  }
	}
  
  return output;
}


//this is sortof map operation where you can insert another argument on top of map

vector<vector<float>> ite2(float (*func)(float,float),vector<vector<float>>v1,vector<vector<float>> v2){
  
  vector<vector<float> >::iterator row1 = v1.begin();
  vector<float>::iterator col1 = row1->begin();
  
  vector<vector<float> >::iterator row2 = v2.begin();
  vector<float>::iterator col2 = row2->begin();
  
  while( row1 != v1.end() && row2 != v2.end() ){
    while( col1 != row1->end() && col2 != row2->end() ){
      *col1 = func(*col1,*col2);
      col1 ++;
      col2 ++;
    }
    row1 ++;
    row2 ++;
  }
  
  
  return v1;
}

float relu(float x){ return x < 0 ? 0 : x; }
float deRelu(float x,float a){ return a < 0 ? 0 : x; }
float sigmoid  (float x){ return x = 1 / ( 1+exp(-x));}
float deSigmoid(float x,float a){ return x * (a * (1 - a));}

float mtanh (float x){ return tanh(x);}
float detanh (float x,float a){ return x*(1 - a*a);}


float zeroize(float x){return 0;}
float add (float x, float y){return x + y ;}

namespace utility {

template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b)
{
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& aVector, const T& aObject)
{
  aVector.push_back(aObject);
  return aVector;
}

}

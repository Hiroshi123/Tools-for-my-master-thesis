#include <iostream>
#include <vector>

class BNN{
  
  vector<vector<float>> N1;
  vector<vector<float>> N2;
  vector<vector<float>> N3;
  
  vector<vector<float>> w1;
  vector<vector<float>> w2;
  
public:
  BNN(int,int,int);
  
  void feedForward(vector<vector<float>>,vector<vector<float>>);
  void calculateErrror();
  void feedBack();
  void updateWeight();
  
};


vector<vector<float>> dot(vector<vector<float>> x,vector<vector<float>> y){
  
  int x_r = x.size();
  int x_c = x[0].size();
  int y_r = y.size();
  int y_c = y[0].size();
  
  vector<vector<float>> z;
  z = vector<vector<float>> (x_r,vector<float>(y_c,0));
  
  for(int i = 0 ; i < x_r ; i++)
    for(int j = 0 ; j < y_c ; j++)
      z[i][j] += x[j][i] * y[i][j];  
  return z;
}



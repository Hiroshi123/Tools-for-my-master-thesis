#include <iostream>
#include <vector>
#include <list>
#include <bitset>

using namespace std;

class BNN{
  
  int inputN,outputN,dataD,batchSize,layerNum;
  float lr = 0.1;
  vector<vector<char>> inputData;
  
  //necessary memory area
  // node (three dimensional for matrix multiplication)
  
  vector<vector<vector<char>>> node;
  
  //vector<vector<vector<float>>>  node;
  vector<vector<vector<char>>> eNode;
  //vector<vector<vector<float>>> eNode;
  
  //teacher signal (two dimensional for dataN & dataDimension)
  vector<vector<char>> teacher;
  
  // variable for storing error propagation
  //vector<vector<float>> E1;
  
  //weight (three dimensional to connect a layer and subsequent layer)
  
  vector<vector<vector<float>>> dw;
  
  vector<vector<vector<float>>> w;
  //vector<vector<float>> w2;
  
public:
  
  BNN(vector<vector<char>>,vector<vector<char>>,vector<int>,int);
  
  vector<vector<float>> randomizeWeight(vector<vector<float>>);
  
  //float relu(float);
  //float deRelu(float,float);
  //vector<vector<float>> ite(vector<vector<float>>);
  
  void feedForward();
  float calculateErrror(int);
  void feedBack();
  void updateWeight();
  void mainLoop(int);
  
};



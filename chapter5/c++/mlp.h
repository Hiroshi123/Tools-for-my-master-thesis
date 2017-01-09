#include <iostream>
#include <vector>
#include <list>

using namespace std;

class BNN{
  
  int inputN,outputN,dataD,batchSize,layerNum;
  float lr = 0.1;
  bool bitCalcOn = false;
  
  int forwardNodeQuantization = 0;
  int errorNodeQuantization = 0;
  int weightQuantization = 0;
    
  vector<vector<float>> inputData;
  
  //necessary memory area
  // node (three dimensional for matrix multiplication)
  
  vector<vector<vector<float>>>  node;
  vector<vector<vector<float>>> eNode;
  
  //teacher signal (two dimensional for dataN & dataDimension)
  vector<vector<int>> teacher;
  
  // variable for storing error propagation
  //vector<vector<float>> E1;
  
  //weight (three dimensional to connect a layer and subsequent layer)
  vector<vector<vector<float>>> dw;
  
  vector<vector<vector<float>>> w;
  //vector<vector<float>> w2;
  
  
public:
  
  BNN(vector<vector<float>>input,vector<vector<int>> teacher,vector<int>modelList,int batchSize,float learningRate,int fnQ,int enQ,int wQ,bool bitCalc);
  vector<vector<float>> randomizeWeight(vector<vector<float>>);
  
  void feedForward();
  float calculateErrror(int);
  void feedBack();
  void updateWeight();
  void mainLoop(int,int);
  void setWeight(std::string,auto);
};



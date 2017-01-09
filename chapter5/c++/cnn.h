#include <iostream>
#include <vector>

using namespace std;

class CNN{
  
  //inout data should be 4 dimension for image processing (Number of data,number of feature, width, height)
  vector<vector<vector<vector<float>>>> inputData;
  
  //teacher data should be 2 dimension (Number of data, 1 of k vector)
  vector<vector<int>> teacherData;
  
  //node should be 5 dimension (layer index, Batch size, feature N, width, height)
  vector<vector<vector<vector<vector<float>>>>> node;
  
  //errorNode should also be 5 dimension (layer index, Batch size, feature N, width, height)
  vector<vector<vector<vector<vector<float>>>>> eNode;
  
  //weight should be 5 dimension (layer index, inputFeatureN, outputFeatureN , kernel width, kernel height)
  vector<vector<vector<vector<vector<float>>>>> weight;

  vector<vector<vector<vector<vector<float>>>>> qWeight;
  vector<vector<vector<vector<vector<float>>>>> qeNode;
  vector<vector<vector<vector<vector<float>>>>> qNode;
  
  vector<pair<char,vector<int>>> model;
  
  int batchSize = 1;
  int kernelW = 3;
  int kernelH = 3;
  float lR = 0.1;
  int layerN;
  bool quantizeN;
  bool quantize;
  
  vector<vector<bool>> binarizeConfig;
  
  vector<float> errorList;
  vector<vector<float>> stock1;//(outputNodeW,vector<float>(outputNodeH,0));
  
public:
  
  CNN(vector<vector<vector<vector<float>>>> input,vector<vector<int>> teacher,vector<pair<char,vector<int>>> m,
      vector<vector<bool>>b,pair<int,int> kS,int batchSize,float learningRate);
  
  void weightInitialize();
  
  void mainLoop(int loopN);
  void dataSetting(int);
  
  void feedForwardBack(bool);
  
  void calculateError(int);
  void updateWeight();
  
  void convolutionStep(int,int wl,bool);
  void activationStep(int,bool);
  void poolingStep(int,bool);
  void fullyConnectedStep(int,int,bool);
  void dimensionShift(int,bool);
  
  //vector<vector<float>> convolution(vector<vector<float>>,vector<vector<float>>);
  //vector<vector<float>>  pooling(vector<vector<float>> data,string type);
  void pooling(int,string);
  vector<vector<float>> rPooling(vector<vector<float>> data,vector<vector<float>> x,string type);

  void errorPlot(void);
  
};


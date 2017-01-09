#include <iostream>
#include <vector>
#include <list>
#include "bnn.h"
#include <math.h>
#include <random>
#include "matrix.cpp"
#include "mnistLoad.cpp"
#include "bnnSub.cpp"
#include <bitset>
#include "bitset2D.h"
#include <string>
#include <boost/utility/binary.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

random_device rd;
//mt19937 mt(rd());
uniform_real_distribution<float> dice(-1.0,1.0);
uniform_int_distribution<int> intDice(0,10);

//this is constructor.. all of initialization is going to be taken place...

BNN::BNN(vector<vector<char>> iData,vector<vector<char>> teacherD, vector<int> layerN,int bS){
  
  this->inputData = iData;
  this->layerNum = layerN.size();
  
  //error aquisition
  
  if(layerN[0] != inputData[0].size()){
    cout << layerN[0] << " " << inputData[0].size() << endl;
    cerr << "dimension of data and actual dimension of input data have to be identical" << endl;
  }
  
  if(layerN[layerNum-1] != teacherD[0].size()){
    cerr << "number of classification and actual dimension of output data have to be identical" << endl;
  }
  
  if(inputData.size() != teacherD.size()){
    cerr << "Number of input data have to be identical with number of output data" << endl;
  }
  
  //parameter settting
  //input data number setting
  
  this->inputN = inputData.size();
  
  //input dimenison setting 
  this->dataD = inputData[0].size();
  
  //output dimension
  this->outputN = layerN[layerNum-1];
  
  //teacher data setting
  this->teacher = teacherD;
  
  //vector<vector<vector<float> > > v
  batchSize = bS;
  
  //node vector memory allocation
  
  for(int i = 0 ; i < layerN.size() ; i++ ){
    int b = layerN[i] / 8;
    if(layerN[i] % 8 != 0)
      b ++;
    
    cout << b << endl;
    
    vector<vector<char>> tempNode(batchSize, vector<char>(b,0));
    node.push_back(tempNode);
    
    if(i != 0){
      
      vector<vector<char>> tempNode(layerN[i-1], vector<char> (layerN[i],0));
      eNode.push_back(tempNode);
      
      vector<vector<float>> tempW(layerN[i-1], vector<float>(layerN[i],0));    
      dw.push_back(tempW);
      w.push_back(tempW);    
      //initial randomization
      this->w[i-1] = randomizeWeight(this->w[i-1]);
    }
  }
  
  cout << node.size() << " : " << node[0].size() << " : " << node[0][0].size() << endl;
  cout << node.size() << " : " << node[1].size() << " : " << node[1][0].size() << endl;
  cout << node.size() << " : " << node[2].size() << " : " << node[2][0].size() << endl;
  
}

vector<vector<float>> BNN::randomizeWeight(vector<vector<float>>weight){
  
  for(int i = 0; i < weight.size() ; i ++)
    for(int j = 0; j < weight[0].size() ; j++)
      weight[i][j] = dice(rd);
  
  return weight;
  
}


void BNN::mainLoop(int epochN){
  
  //stochastic gradient dicent....
  //pick up one of data among dataset

  std::cout << epochN << std::endl;
  
  int error = 0;
  int dataIndex;
  
  for(int k = 0; k < epochN; k ++){
    if(k % 10 == 0){
      cout << "error : " << error << endl;
      error = 0;
    }
    
    //i = rand() % inputN;
    dataIndex = rand() % 100;
    
    for(int j = 0 ; j < this->dataD ; j++)
      for(int b = 0 ; b < this->batchSize ; b++)
	this->node[0][b][j] = this->inputData[dataIndex+b][j]; 
    
    feedForward();
    
    //calculateErrror(dataIndex);
    
    error += calculateErrror(dataIndex);
    
    feedBack();
    updateWeight(); 
  }  
}

float quantization(float x) { return x > 0 ? 1 : -1 ;}
float quantizationN(float x) { return x > 0 ? 1 : 0 ;}
float quantizationN2(float x,float a) { return a > 0 && x > 0 ? 1 : 0 ;}

void BNN::feedForward()
{
  //k denotes kth layer
  for(int k = 0 ; k < layerNum - 1 ; k ++){
    //first, you quantize weights to either pos 1 oder neg 1
    vector<vector<float>>wT = ite(quantization,this->w[k]);
    //second, you convert float value into character type
    //the value is going to be changed into either 1 or 0
    vector<vector<char>> s = inputBinarize(transpose(wT));
    
    bitset<8> tempBit1,tempBit2;
    char bitSave[8];
    int outputNodeN =   s.size();
    int inputNodeN  = s[0].size();
    for(int j = 0 ; j < outputNodeN ; j ++ ){
      int bitStock1,bitStock2 = 0;
      for(int i = 0 ; i < inputNodeN ; i ++ ){
        tempBit1 = (s[j][i] & node[k][0][i]);  //bit And operation
        tempBit2 = ((~s[j][i]) & node[k][0][i]);  //bit And operation after flipping (to simulate negative weight)
        bitStock1 += tempBit1.count();
        bitStock2 += tempBit2.count();
      }
      //cout << bitStock1 << " : " << bitStock2 << endl;
      bitSave[j%8] = bitStock1 > bitStock2 ? '1' : '0';
      //cout << bitSave[j%8] << endl;
      if(j % 8 == 7 || j == s.size()-1)
        node[1][0][j/8] = strToChar(bitSave);
    }
  }
}


float BNN::calculateErrror(int dataIndex){
  
  //difference between values of end layer and
  
  int errorSigma = 0;
  int temp = 0;
  
  //iterate in the dimension of data
  
  for(int i = 0 ; i < 2 ; i++)
    for(int b = 0 ; b < this->batchSize ; b++){
      cout << "diff" << endl;
      printBinary(this->teacher[dataIndex][i]);
      printBinary(this->node[this->layerNum-1][b][i]);
      bitset<8> tBit;
      tBit = (this->teacher[dataIndex][i] ^ this->node[this->layerNum-1][b][i]);
      eNode[this->layerNum-2][b][i] = strToChar(tBit.to_string().c_str());
      errorSigma += tBit.count();
    }
  
  cout << errorSigma << endl;
  
  return errorSigma;
  
}

void BNN::feedBack(){
  
  for(int k = this->layerNum - 3; k >= 0 ; k--){  
    vector<vector<float>>wT = ite(quantization,this->w[k]);
    //second, you convert float value into character type
    //the value is going to be changed into either 1 or 0
    vector<vector<char>> s = inputBinarize(transpose(wT));
    bitset<8> tempBit1,tempBit2;
    char bitSave[8];
    vector<char> tNode;
    int outputNodeN =   s.size();
    int inputNodeN  = s[0].size();
    for(int j = 0 ; j < outputNodeN ; j ++ ){
      int bitStock1,bitStock2 = 0;
      for(int i = 0 ; i < inputNodeN ; i ++ ){
        tempBit1 = (s[j][i] & eNode[k][0][i]);  //bit And operation
	tempBit2 = ((~s[j][i]) & eNode[k][0][i]);  //bit And operation after flipping (to simulate negative weight)
        //cout << tempBit.count() ;
        bitStock1 += tempBit1.count();
        bitStock2 += tempBit2.count();
      }
      //cout << bitStock1 << " : " << bitStock2 << endl;
      bitSave[j%8] = (bitStock1 > bitStock2) & this->node[k+1][0][j] ? '1' : '0';
      //cout << bitSave[j%8] << endl;
      if(j % 8 == 7 || j == s.size()-1)
        node[1][0][j/8] = strToChar(bitSave);
    }
  }
  
}


void BNN::updateWeight() {
  
  for(int i = 0; i < this->layerNum - 1 ; i++)
    dw[i] = dot(transpose(this->node[i]),this->eNode[i]);
  
  //auto f1 = [](float x,y) { return this->lr * y + x };
  
  for(int k = 0 ; k < w.size() ; k++)
    for(int i = 0 ; i < w[k].size() ; i++)
      for(int j = 0 ; j < w[k][i].size() ; j++)
	w[k][i][j] += 0.1* dw[k][i][j];
  
}

vector<vector<float>> prepareData(vector<vector<float>> input){
  
  for(int i = 0;i < input.size(); i++)
    for(int j = 0;j < input[0].size(); j++)
      input[i][j] = dice(rd);
  
  return input;
}

vector<vector<int>> prepareTeacher(vector<vector<int>> t){
  
  for(int i = 0;i<t.size();i++)
    for(int j = 0;j<t[0].size();j++)
      t[i][j] = 0;
  
  int outputClass = t[0].size();
  int s;
  for(int i = 0;i<t.size();i++){
    s = rand() % outputClass;
    t[i][s] = 1;
  }
  return t;    
}


int main(int argc,char* argv[]){
  
  short bin = BOOST_BINARY(1000);
  
  Mnist mnist;
  
  vector<vector<float>> mnistImages;
  vector<vector<int>> mnistTeacher;
  
  mnistImages  = mnist.readTrainingFile("../data/mnist/train-images-idx3-ubyte");
  mnistTeacher = mnist.readLabelFile("../data/mnist/train-labels-idx1-ubyte");
  
  cout << mnistImages.size() << ":" << mnistImages[0].size() << endl;
  cout << mnistTeacher.size() << " : " << mnistTeacher[0].size() << endl;
  
  int dataN   = mnistImages.size();
  int dataD   = mnistImages[0].size();
  int outputD = mnistTeacher[0].size();
  
  vector<int> layer = {dataD,500,10};
  
  int batchSize = 1;
  
  vector<vector<char>> mnistBinaryInputData;
  vector<vector<char>> mnistBinaryTeacherData;
  
  mnistBinaryInputData = inputBinarize(mnistImages);
  mnistBinaryTeacherData = inputBinarize(mnistTeacher);
  
  cout << mnistBinaryInputData.size() << " : " << mnistBinaryInputData[0].size() << endl;
  cout << mnistBinaryTeacherData.size() << " : " << mnistBinaryTeacherData[0].size() << endl;
  for(int i = 0; i < 10; i ++)
    cout << mnistTeacher[0][i];
  
  cout << " " << endl;
  printBinary(mnistBinaryTeacherData[0][0]);
  printBinary(mnistBinaryTeacherData[0][1]);
  
  BNN bnn(mnistBinaryInputData,mnistBinaryTeacherData,layer,batchSize);
  
  bnn.mainLoop(1);
  
  cout << "a" << endl;  
  
}


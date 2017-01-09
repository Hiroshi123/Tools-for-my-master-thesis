#include <iostream>
#include <vector>
#include <list>
#include <math.h>
#include <random>
//#include <bitset> 
//#include "bitset2D.h" 
#include <string.h> //for strcmp
#include <unistd.h>
#include <boost/dynamic_bitset.hpp>

//following are files on the same directory
#include "mlp.h"
#include "lib/matrix.cpp"
#include "lib/mnistLoad.cpp"
#include "lib/bnnSub.cpp"

using namespace std;

random_device rd;
//if you let generator mercenn twister, uncomment a following line.
//mt19937 mt(rd());
uniform_real_distribution<float> dice(-1.0,1.0);
uniform_int_distribution<int> intDice(0,10);


//this is constructor.. all of initialization is going to be taken place...

BNN::BNN(vector<vector<float>> iData,vector<vector<int>> teacherD, vector<int> layerN,int bS,float learnR,int fnQ,int enQ,int bwQ,bool bitCalc){
  
  this->inputData = iData;
  this->layerNum = layerN.size();
  this->lr = learnR;
  
  this->forwardNodeQuantization = fnQ;
  this->errorNodeQuantization = enQ;
  this->weightQuantization = bwQ;
  this->bitCalcOn = bitCalc;
  
  //error aquisition
  
  if(layerN[0] != inputData[0].size()){
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
  
  batchSize = bS;
  
  //node vector memory allocation
  
  for(int i = 0 ; i < layerN.size() ; i++ ){
    vector<vector<float>> tempNode(batchSize, vector<float>(layerN[i],0));
    node.push_back(tempNode);
    
    if(i != 0) {
      
      vector<vector<float>> tempNode(batchSize, vector<float>(layerN[i],0));
      eNode.push_back(tempNode);
      vector<vector<float>> tempW(layerN[i-1], vector<float>(layerN[i],0));
      
      dw.push_back(tempW);
      w.push_back(tempW);
      
      // if( i == 1){
      // 	setWeight("./data/02/weightB1.bin",&w[i-1]);
      // } else if (i == 2){
      // 	setWeight("./data/02/weightB2.bin",&w[i-1]);
      // }
      
      //initial randomization
      
      this->w[i-1] = randomizeWeight(this->w[i-1]);
      
    }
  }
}


void BNN::setWeight(std::string path,auto x){
  
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  
  if(!fin) {
    std::cout << "not able to open!" << std::endl;
    //return 1;
  }
  
  std::vector<int> tempV;
  int n;
  while(!fin.eof()){
    fin.read((char*)&n, sizeof(int));
    tempV.push_back(n);
  }
  
  std::cout << tempV.size() << std::endl;
  for(int i = 0 ; i < (*x).size() ; i++){
    for(int j = 0 ; j < (*x)[i].size() ; j++ )
      (*x)[i][j] = tempV[i*500+j];
  }
  
};

vector<vector<float>> BNN::randomizeWeight(vector<vector<float>>weight){
  
  for(int i = 0; i < weight.size() ; i ++)
    for(int j = 0; j < weight[0].size() ; j++){
      weight[i][j] = quantization(dice(rd));
    }
  
  
  return weight;
}

void BNN::mainLoop(int epochN,int dataRange){
  
  //stochastic gradient dicent....
  //pick up one of data among dataset
  
  float error = 0;
  int dataIndex;
  
  epochN = 10000;
  
  for(int k = 0; k < epochN; k ++){
    if(k % 10 == 0){
      cout << "error : " << error << endl;
      error = 0;
    }
    
    //i = rand() % inputN;
    dataIndex = rand() % dataRange;
    
    for(int j = 0 ; j < this->dataD ; j++)
      for(int b = 0 ; b < this->batchSize ; b++)
	this->node[0][b][j] = this->inputData[dataIndex+b][j];
    
    //std::cout << std::endl;
    //std::for_each(this->node[0][0].begin(),this->node[0][0].end(),[](float v){std::cout << v;});
    //std::cout << std::endl;
    
    feedForward();
    
    error += calculateErrror(dataIndex);
    
    feedBack();
    updateWeight();
    
  }
  
  std::cout << this->w.size() << std::endl;
  
  vector<vector<float>> wT1 = ite(quantization,this->w[0]);
  vector<vector<float>> wT2 = ite(quantization,this->w[1]);
  
  
  std::cout << wT1.size() << " " << wT1[0].size() << std::endl;
  std::cout << wT2.size() << " " << wT2[0].size() << std::endl;
  
  // std::cout << std::endl;
  
  std::vector<std::vector<int>> w1(w[0].size(),std::vector<int>(w[0][0].size()));
  std::vector<std::vector<int>> w2(w[1].size(),std::vector<int>(w[1][0].size()));
  
  for(int i = 0 ;  i < 2 ; i++)
    for(int j = 0 ;  j < w[i].size() ; j++){
      int s = 0;
      std::for_each(w[i][j].begin(),w[i][j].end(),[&](float v){
	  if(i == 0)
	    w1[j][s] = (quants(v));
	  else
	    w2[j][s] = (quants(v));
	  s++;
	});
    }

  int len  = w1.size()*w1[0].size();
  int len2 = w2.size()*w2[0].size();
  
  int buffer1[len];
  int buffer2[len2];
  
  //copy
  for(int i = 0 ; i < w1.size() ; i++ )
    for(int j = 0 ; j < w1[0].size() ; j++ )
      buffer1[i*w1[0].size()+j] = w1[i][j];
  
  for(int i = 0 ; i < w2.size() ; i++ )
    for(int j = 0 ; j < w2[0].size() ; j++ )
      buffer2[i*w2[0].size()+j] = w2[i][j];
  
  std::ofstream fout;
  fout.open("weightB1.bin",ios::out|ios::binary|ios::trunc);
  
  if(!fout){
    std::cout << "the file could not be opened..";
  } else {
    for(int i = 0 ; i < len ; i++ )
      fout.write((const char*) &buffer1[i],sizeof(int));
  }
  
  fout.close();
  
  //fout.close();
  std::ofstream fout2;
  fout2.open("weightB2.bin",ios::out|ios::binary|ios::trunc);
  
  if(!fout){
    std::cout << "the file could not be opened..";
  } else {
    for(int i = 0 ; i < len2 ; i++ )
      fout2.write((const char*) &buffer2[i],sizeof(int));
  }
  
  fout2.close();
  
  
  
}


void BNN::feedForward()
{
  
  for(int i = 0; i < this->layerNum-1 ; i++){
    
    vector<vector<float>> nT;
    vector<vector<float>> wT;
    
    //vector<vector<float>> temp1;
    //vector<vector<float>> temp2;
    
    //node quantization
    switch(this->forwardNodeQuantization){
    case 0:
      //0 means as it is
      nT = this->node[i];
      break;
    case 1:
      //quantization to either 0 or 1;
      //nT = ite(quantizationN,this->node[i]);

      if( i == 0 ){
	nT = ite(quantizationNS,this->node[i]);
      } else {
	nT = ite(quantization,this->node[i]);
      }
      
      break;
    }
    //weight quantization
    switch(this->weightQuantization){
    case 0:
      //0 means as it is
      wT = this->w[i];
      break;
    case 1:
      //quantization to either 1 or -1
      wT = ite(quantization,this->w[i]);
      break;
    case 2:
      //quantization to either 1, 0, or -1
      wT = ite(ternalization,this->w[i]);
      break;
    }
    
    bool forward = true;
    
    if(this->bitCalcOn){
      
      this->node[i+1] = ite ( quantization, bitDot(nT,wT,false));
      //this->node[i+1] = ite ( quantization, bitDot(nT,wT,false));
      //this->node[i+1] = ite ( mtanh, bitDot(nT,wT,false));
      //this->node[i+1] = ite ( sigmoid, bitDot(nT,wT,forward));
      
    } else {
	this->node[i+1] = ite ( quantization, dot(nT,wT));
	//this->node[i+1] = ite ( sigmoid, dot(nT,wT));
	
    }
    
    //std::cout << this->node[i+1][0][0] << std::endl;
    
  }
  
}

float BNN::calculateErrror(int dataIndex){
  
  //difference between values of end layer and
  
  float errorSigma = 0.0;
  float temp = 0.0;
  
  //iterate in the dimension of data
  
  for(int i = 0 ; i < this->outputN ; i++)
    for(int b = 0 ; b < this->batchSize ; b++){
      
      //std::cout << this->node[this->layerNum-1][b][i];
      this->eNode[this->layerNum-2][b][i] = this->teacher[dataIndex][i] - this->node[this->layerNum-1][b][i];
      
      temp = pow(this->eNode[this->layerNum-2][b][i],2);
      errorSigma += temp;
      //std::cout << temp << std::endl;
      
    }
  
  //cout << errorSigma << endl;
  return errorSigma;
  
}

void BNN::feedBack(){
  
  for(int i = this->layerNum - 3; i >= 0 ; i--){
    //this->w[i] = ite(quantization,this->w[i]);
    
    vector<vector<float>> nT,wT;
    
    switch(this->errorNodeQuantization){
    case 0:
      // 0 is as it is
      nT = this->eNode[i+1];
      break;
      
    case 1:
      // note here quantization is either 1 or -1 since backward pass is not limited in positive range
      //nT = ite(quantization,this->eNode[i+1]);
      nT = ite(ternalization,this->eNode[i+1]);
      break;
    case 2:
      // quantization to 1 , 0 ,or -1 
      nT = ite(ternalization,this->eNode[i+1]);
      break;
    }
    switch(this->weightQuantization){
    case 0:
      // 0 is as it is
      wT = this->w[i+1];
      break;
    case 1:
      //quantization to either 1 or -1
      wT = ite(quantization,this->w[i+1]);
      break;
    case 2:
      //trinalization to either 1, 0, or -1
      wT = ite(ternalization,this->w[i+1]);
      break;
    }
    
    bool forward = false;
    
    //std::cout << "fk " << i << std::endl;
    
    //std::cout << this->eNode[i][0][0] << std::endl;
    //std::cout << this->eNode[i+1][0][0] << std::endl;
    //std::cout << "nT " << nT[0][0] << std::endl;
    
    if(this->bitCalcOn)
      //this->eNode[i] = bitDot(nT,transpose(wT),forward);
      this->eNode[i] = ite2( qf1, bitDot(nT,transpose(wT),forward),node[i+1]);
      //this->eNode[i] = ite2( deSigmoid, bitDot(nT,transpose(wT),forward),node[i+1]);
      //this->eNode[i] = ite2( qf1, dot(nT,transpose(wT)),node[i+1]);
      //this->eNode[i] = ite2( detanh, bitDot(nT,transpose(wT),forward),node[i+1]);
    
    
    else
      this->eNode[i] = ite2( quantization2, dot(nT,transpose(wT)),node[i+1]);
    
    //this->eNode[i] = ite2( deSigmoid, dot(nT,transpose(wT)),node[i+1]);
      //this->eNode[i] = ite2( detanh, bitDot(nT,transpose(wT),forward),node[i+1]);
    
    
    //std::for_each(this->eNode[i].begin(),this->eNode[i][0].end(),[](float v){std::cout << v << ",";});
    //std::cout << std::endl;
    
    //std::cout << this->eNode[i][0][0] << std::endl;
    
  }
}

void BNN::updateWeight() {
  
  for(int i = 0; i < this->layerNum - 1 ; i++){
    //std::for_each(this->eNode[i][0].begin(),this->eNode[i][0].end(),[](float v){std::cout << v << ",";});
    if(i==0){
      //std::cout << this->eNode[i][0][5] << std::endl;
      dw[i] = dot(ite(quantizationNS,transpose(this->node[i])),this->eNode[i]);
    }
      
    else
      dw[i] = dot(transpose(this->node[i]),this->eNode[i]);
    
    
    //if(i==1)
    //  std::for_each(this->eNode[i][0].begin(),this->eNode[i][0].end(),[](float v){std::cout << v << ",";})
    //std::for_each(this->dw[i][2].begin(),this->dw[i][2].end(),[](float v){std::cout << v << ",";});
  }
  
  // for(int i = 0 ; i < w[1].size() ; i++)
  //   for(int j = 0 ; j < w[1][i].size() ; j++){
  //     if(dw[1][i][j] > 0){
  // 	std::cout << w[1][i][j] << std::endl;
  // 	w[1][i][j] = 0.1;
  //     } else if(dw[1][i][j] < 0) {
  // 	std::cout << w[1][i][j] << std::endl;
  // 	//w[1][i][j] = -1;
  // 	w[1][i][j] = -0.1;
  //     } else {
	
  //     }
  //   }
  
  
  //w[1][i][j] += this->lr * dw[k][i][j];
    
  // for(int k = 0 ; k < w.size() ; k++)
  //   for(int i = 0 ; i < w[k].size() ; i++)
  //     for(int j = 0 ; j < w[k][i].size() ; j++)
  // 	w[k][i][j] += this->lr * dw[k][i][j];

  int maxE = 15;

  for(int k = 0 ; k < w.size() ; k++)
    for(int i = 0 ; i < w[k].size() ; i++)
      for(int j = 0 ; j < w[k][i].size() ; j++){
	//std::cout << w[1][i][j] << " " << dw[1][i][j] << std::endl;
	if(w[k][i][j] >= maxE || w[k][i][j] <= -maxE){}
	else {
	  w[k][i][j] += this->lr * dw[k][i][j];
	}
      }
  
  
  //w[1][i][j] += dw[1][i][j];
  
}

int main(int argc,char* argv[]){
  
  //**************************************************//
  //following is a procedure to load Mnist dataset

  std::cout << "start" << std::endl;
  
  Mnist mnist;
  
  vector<vector<float>> mnistImages;
  vector<vector<int>> mnistTeacher;
  
  mnistImages  = mnist.readTrainingFile("../../data/mnist/train-images-idx3-ubyte");
  mnistTeacher = mnist.readLabelFile("../../data/mnist/train-labels-idx1-ubyte");
  
  //cout << mnistImages.size() << ":" << mnistImages[0].size() << endl;
  //cout << mnistTeacher.size() << " : " << mnistTeacher[0].size() << endl;
  
  int dataN   = mnistImages.size();
  int dataD   = mnistImages[0].size();
  int outputD = mnistTeacher[0].size();
  
  std::cout << mnistTeacher.size() << "," << mnistTeacher[0].size() << std::endl;
  for(int i = 0; i < mnistTeacher.size() ; i++ ){
    for(int j = 0 ; j < mnistTeacher[i].size() ; j++ ){
      if(mnistTeacher[i][j] == 0 )
	mnistTeacher[i][j] = -1;
    }
  }
  
  //**************************************************//
  
  //**************************************************//
  
  //following is a parameter setting where you can pick up arbitrarily.
  
  //set network architecture here
  
  vector<int> layer = {dataD,500,10};
  float learningRate = 0.1;
  
  //since Mnist contains training set in a randomized manner, it can be decently trained when you train with only nth low index data.
  
  int dataRange = 500; //maximum 60000
  
  //batch size have to be less than dataRange.
  
  int batchSize = 1;
  int iteration = 1000;
  
  int bitCalc = true;
  
  //*************************************************//
  
  //arguments of command should be set for configuration of binarization setting.
  
  int result;
  
  int fnQ = 0;
  int enQ = 0;
  int wQ  = 0;
  
  while((result=getopt(argc,argv,"f:e:w:"))!=-1){
    switch(result){
    case 'f':
      if(optarg[0] == 'x')
	fnQ = 0;
      else if(optarg[0] == 'b')
	fnQ = 1;
      else
	fnQ = 0;
      break;
    case 'e':
      if(optarg[0] == 'x')
	enQ = 0;
      else if(optarg[0] == 'b')
	enQ = 1;
      else if(optarg[0] == 't')
	enQ = 2;
      else
	enQ = 0;
      break;
    case 'w':
      if(optarg[0] == 'x')
	wQ = 0;
      else if(optarg[0] == 'b')
	wQ = 1;
      else if(optarg[0] == 't')
	wQ = 2;
      else
	wQ = 0;
      break;
    case ':':
      fprintf(stdout,"%c needs value\n",result);
      break;
    case '?':
      fprintf(stdout,"unknown\n");
      break;
    }
  }
  
  cout << fnQ << " : " << enQ << " : " << wQ << endl;
  
  BNN bnn(mnistImages,mnistTeacher,layer,batchSize,learningRate,fnQ,enQ,wQ,bitCalc);
  
  bnn.mainLoop(iteration,dataRange);
  
}


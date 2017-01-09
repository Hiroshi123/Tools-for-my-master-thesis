#include <iostream>
#include <vector>
#include <list>
//#include "bnn.h"
#include <math.h>
#include <random>
#include <bitset>
#include <map>
#include <tuple>
#include <algorithm>

#include "matplotlibcpp.h"
#include "cnn.h"
#include "lib/matrix.cpp"
#include "lib/mnistLoad.cpp"
#include "lib/loadCifar.cpp"
#include "lib/bnnSub.cpp"

//#include "bitset2D.h"

using namespace std;
namespace plt = matplotlibcpp;

random_device rd;
//mt19937 mt(rd());
uniform_real_distribution<float> dice(-1.0,1.0);
uniform_int_distribution<int> intDice(0,10000);


CNN::CNN(vector<vector<vector<vector<float>>>> input,vector<vector<int>> teacher,vector<pair<char,vector<int>>> m,
	 vector<vector<bool>>b,pair<int,int> ks,int bS,float learnR){
  
  cout << input.size() << " " << input[0].size() << " " << input[0][0].size() << endl;
  
  this->inputData   = input;
  this->teacherData = teacher;
  this->model = m;
  this->batchSize = bS;
  
  this->kernelW = ks.first;
  this->kernelH = ks.second;
  this->batchSize = bS;
  this->lR = learnR;
  
  this->layerN = this->model.size();
  this->quantize  = true ;//true;
  this->quantizeN = true ;//true;
  
  this->binarizeConfig = b;
  
  //**********************************************************************************************************************//
  //error handling for configuration of model setting
  
  char operation;
  
  for(int i = 0 ; i < this->model.size() ; i++){
    
    operation = this->model[i].first;
    
    if (operation == 'c' || operation == 'p'){
      if(this->model[i].second.size() != 3)
	cerr << "you must set 3 dimensional vector for convolution and pooling layer" << endl;
      assert(this->model[i].second.size() == 3);
    }
    
    if (operation == 'f' || operation == 't' || operation == 's' ){
      if(this->model[i].second.size() != 1)
	cerr << "you must set 1 dimensional vector for convolution and pooling layer must be " << endl;
      assert(this->model[i].second.size() == 1);
    }
    
    if(operation == 'c'){
      if(
	 (this->model[i-1].second[1] != this->model[i].second[1]) ||
	 (this->model[i-1].second[2] != this->model[i].second[2])
	 )
	cerr << "In between convolution operation, data size should be identical." << endl;
      assert(this->model[i-1].second[1] == this->model[i].second[1]);
      assert(this->model[i-1].second[2] == this->model[i].second[2]);
    }
    
    if(this->model[i].first == 'p'){
      if( this->model[i-1].second[1] != (this->model[i].second[1] * 2) ){
	cerr << "In between pooling layer, resolution have to turn into half." << endl;
	cerr << "But, subsequent layer " << i << " : " << this->model[i].second[1] <<
	  " is not as half as layer " << i - 1 << " : " << this->model[i-1].second[1] << endl;
      }
      assert(this->model[i-1].second[1] == (this->model[i].second[1] * 2));
    }
    
    //**********************************************************************************************************************//
    
    //**********************************************************************************************************************//
    //memory allocation for nodes, nodes for error, and weights on each layers
    
    
    //if you are on the model which is fully connected layer or last layer on which teacher node will sit, dimension of node and weight is going
    //to be negative -2 from definition of model.
    //specifically, model is going to be two dimension which contains only ( 1,1,batchSize, dimension), first & second index is going to be ignored.
    
    if(operation == 's' || operation == 'f' || operation == 't'){
      
      int featureN = this->model[i].second[0];
      
      //node setting
      vector<vector<vector<vector<float>>>> tN(1,vector<vector<vector<float>>>(1,vector<vector<float>>(batchSize,vector<float>(featureN,0) )));
      node.push_back(tN);
      qNode.push_back(tN);
      //As of error node setting and weight setting, first node should be omitted.
      if(i != 0){
	
	//error node setting
	vector<vector<vector<vector<float>>>> teN(1,vector<vector<vector<float>>>(1,vector<vector<float>>(batchSize,vector<float>(featureN,0) )));
        eNode.push_back(teN);
	qeNode.push_back(teN);
	//weight setting
	//no weight setting for transition layer from 4D to 2D
	if(operation != 's'){
	  int preFeN = this->model[i-1].second[0];
	  vector<vector<vector<vector<float>>>> tW(1,vector<vector<vector<float>>>(1,vector<vector<float>>(preFeN,vector<float>(featureN,0) )));
	  weight.push_back(tW);
	  qWeight.push_back(tW);
	}
      }
      
    } else {
      
      // follwing is you are allocating memory for convolution or pooling layer.
      int featureN = this->model[i].second[0];
      int nodeW    = this->model[i].second[1];
      int nodeH    = this->model[i].second[2];
      
      //cout << i << " " << featureN << " : " <<  nodeW << "  " << nodeH << endl;
      
      //node setting
      vector<vector<vector<vector<float>>>> tN(this->batchSize,vector<vector<vector<float>>>(featureN,vector<vector<float>>(nodeW,vector<float>(nodeH,0) )));
      node.push_back(tN);
      qNode.push_back(tN);
      
      //cout << featureN << " : " << node[i][0][0].size() << " : " << node[i][0][0][0].size() << endl;
      
      if(i != 0){
	//memory allocation for errorNode(node for backpropagation)
        vector<vector<vector<vector<float>>>> enT (batchSize,vector<vector<vector<float>>>(featureN,vector<vector<float>>(nodeW,vector<float>(nodeH,0) )));
	eNode.push_back(enT);
	qeNode.push_back(enT);
	
        int preFeN = this->model[i-1].second[0];
	//weight setting which came from a model
	//note nothing for pooling step
	if(operation == 'c'){
	  
	  vector<vector<vector<vector<float>>>> tW(preFeN,vector<vector<vector<float>>>(featureN,vector<vector<float>>(kernelW,vector<float>(kernelH,0) )));
	  weight.push_back(tW);
	  qWeight.push_back(tW);
	}
      }
    } 
  }
  
  // Error aquisition in case non-identicality between vector of weight binarization configuration and actual vector which contains weight
  for(int i = 0; i < this->binarizeConfig.size(); i ++ ){
    if(this->weight.size() != this->binarizeConfig[i].size()){
      cout << this->weight.size() << " : " << this->binarizeConfig[i].size() << endl;
      cerr << " The size of vector for binarization setting have to be equivalent to the layer of weight.." << endl;
    }
    assert(this->weight.size() == this->binarizeConfig[i].size());
  }
  
  //**********************************************************************************************************************//
  
  
};



void CNN::weightInitialize(){
  
  //since vector is 5 dimension for cnn for image
  auto func = [](float x) {return dice(rd); };
  
  for(int i = 0; i < weight.size() ; i ++)
    weight[i] = ite4(func,weight[i]);
  
}

void CNN::mainLoop(int iterationM){
  
  weightInitialize();
  int dd = 0;
  for(int d = 0 ; d < iterationM; d++ ){
    
    dd = intDice(rd);
    dataSetting(dd);
    feedForwardBack(true);
    calculateError(dd);
    feedForwardBack(false);
    updateWeight();
    
  }
  
}

void CNN::calculateError(int d){
  
  float errorSigma = 0;
  float temp;
  
  for(int b = 0; b < this->batchSize; b++ )
    for(int o = 0 ; o < this->teacherData[0].size() ; o++ ){
      temp = this->teacherData[d+b][o] - this->node[layerN-1][0][0][b][o];
      this->eNode[layerN-2][0][0][b][o] = temp;
      errorSigma += temp * temp;
      //cout << temp << " ";
      //cout << this->teacherData[d][0] << " : " << this->node[layerN-1][d][0][0][o] << endl;
    }
  
  this->errorList.push_back(errorSigma);
  cout << "error: " << errorSigma << endl;
  
}

void CNN::errorPlot(){
  
  plt::title("error transition");
  plt::xlabel("Epoch times");
  plt::ylabel("error");
  plt::plot(this->errorList);
  plt::show();
}

void CNN::updateWeight(){
  
  vector<vector<float>> stockA;//(kernelW,vector<float>(kernelH,0));
  vector<vector<float>> dt;
  vector<vector<float>> dt2;
  
  int wIndex = 0;
  
  //cout << "; " << model.size() << endl;
  
  for(int l = 0 ; l < this->model.size() ; l ++ ){
    
    char operation = this->model[l].first;
    if( operation == 'c' || operation == 'f' || operation == 't' ){
      
      int outputFeatureN = this->eNode[l-1][0].size();
      int inputFeatureN  = this->node[l-1][0].size();
      
      int nodeW = this->node[l-1][0][0].size();
      int nodeH = this->node[l-1][0][0][0].size();
      
      //cout << outputFeatureN << inputFeatureN << endl;
      //cout << nodeW << " " << nodeH << endl;
      
      //first iteration is
      //if(operation == 'c')
      for(int b = 0; b < batchSize; b++)
	for(int o = 0 ; o < outputFeatureN ; o++ ){
	  for(int i = 0 ; i < inputFeatureN ; i++ ){
	    
	    if(operation == 'c'){
	      stockA = vector<vector<float>> (kernelW,vector<float>(kernelH,0));
	      for(int w = 0 ; w < nodeW ; w ++){
		for(int h = 0 ; h < nodeH ; h ++){
		  for(int kw = 0 ; kw < this->kernelW ; kw ++ )
		    for(int kh = 0 ; kh < this->kernelH ; kh ++ ){
		      if ( 0 <= w+kw-1 && w+kw-1 < nodeW &&
			   0 <= h+kh-1 && h+kh-1 < nodeH
			   )
			stockA[kw][kh] += node[l-1][b][i][w][h] * eNode[l-1][b][o][w+kw-1][h+kh-1];
		    }
		}
	      }
	      
	      for(int kw = 0 ; kw < this->kernelW ; kw ++ )
		for(int kh = 0 ; kh < this->kernelH ; kh ++ ){
		//cout << stock[kw][kh] << " ";
		//		cout << stock[kw][kh] << endl;
		//this->weight[wIndex][i][o][kw][kh] += 0.1 * stockA[kw][kh];
		stockA[kw][kh] = 0;
	      }
	    }
	    
	    if(operation == 'f' || operation == 't'){
	      
	      int inputN  = this->node[l-1][0][0][0].size();
	      int outputN = this->node[l][0][0][0].size();
	      
	      dt = vector<vector<float>> (inputN,vector<float>(outputN,0));
	      dt = dot(transpose(node[l-1][0][0]) , eNode[l-1][0][0]);
	      
	      for(int kw = 0 ; kw < inputN ; kw ++ )
		for(int kh = 0 ; kh < outputN ; kh ++ ){
		  this->weight[wIndex][0][0][kw][kh] += 0.1 * dt[kw][kh];
		  //dt[kw][kh] = 0;
		}
	      
	    }
	    
	  }
	}
      
      wIndex ++;
      
    }
  }
  
  //for debug information
  /*
  for(int l = 0 ; l < weight.size() ; l++ ){
    cout << "layer"  << l << endl;
    for(int i = 0 ; i < weight[l].size() ; i++ )
      for(int o = 0 ; o < weight[l][i].size() ; o++ )
	for(int w = 0 ; w < weight[l][i][o].size() ; w++ )
	  for(int h = 0 ; h < weight[l][i][o][w].size() ; h++ ){
	    if(this->weight[l][i][o][w][h] > 0)
	      cout << this->weight[l][i][o][w][h] << " ";
            else if (this->weight[l][i][o][w][h] == 0)
	      cout << 0 << " ";
	    else
	      cout << this->weight[l][i][o][w][h] << " ";
	  }
   
  }
  */
  
}


void CNN::dataSetting(int d){
  
  for(int b = 0 ; b < this->batchSize ; b++ ){
    this->node[0][b] = this->inputData[d+b];
  }
  
}


void CNN::feedForwardBack(bool forward){
  
  // CNN contains multiple different steps e.g. convolution, pooling, connection from convolution part to fully connected part,
  // , and fully connected layer.
  //cout << this->model.size() << endl;
  
  //outmost loop is for different layers
  
  int dir = (forward) ? 1 : -1;
  int cor = (forward) ? 1 :  0;
  int l   = (forward) ? 0 : layerN-2;
  int wl  = (forward) ? 0 : weight.size()-1; //index for weight
  
  while( true ){
    
    //cout << l << endl;
    
    if(this->model[l+1].first == 'c'){
      
      //this->node[l]   = ite4(quantization,this->node[l]);
      
      this->qNode[l]    = (this->binarizeConfig[0][wl]) ? ite4(quantizationN,this->node[l]) : this->node[l];
      this->qeNode[l]   = (this->binarizeConfig[1][wl]) ? ite4(quantization,this->eNode[l]) : this->eNode[l];
      this->qWeight[wl] = (this->binarizeConfig[2][wl]) ? ite4(quantization,this->weight[wl]) : this->weight[wl];	
      
      convolutionStep(l,wl,forward);
      activationStep (l+dir,forward);
      wl += dir;
      
    }
    
    else if(this->model[l+1].first == 'p'){
      //cout << "miss" << endl;
      poolingStep(l,forward);
    }
    
    else if(this->model[l+1].first == 's'){
      
      dimensionShift(l,forward);
    }
    
    else if(this->model[l+1].first == 'f' || this->model[l+1].first == 't'){

      this->qNode[l]    = (this->binarizeConfig[0][wl]) ? ite4(quantizationN,this->node[l]) : this->node[l];
      this->qeNode[l]   = (this->binarizeConfig[1][wl]) ? ite4(quantization,this->eNode[l]) : this->eNode[l];
      this->qWeight[wl] = (this->binarizeConfig[2][wl]) ? ite4(quantization,this->weight[wl]) : this->weight[wl];	
      
      /*
      if(quantizeN && forward){
	//	this->qNode = vector<vector<vector<vector<float>>>>
	//  (this->node[l].size(),vector<vector<vector<float>>>(this->node[l][0].size(),vector<vector<float>>(this->node[l][0][0].size(),vector<float>(this->node[l][0][0][0].size(),0))));
	//this->qNode = this->node[l];
	this->qNode[l] = ite4(quantizationN,this->node[l]);
	}
      
      if(quantizeN && !forward){
	//this->qNode = vector<vector<vector<vector<float>>>>
	//  (this->node[l+1].size(),vector<vector<vector<float>>>(this->node[l+1][0].size(),vector<vector<float>>(this->node[l+1][0][0].size(),vector<float>(this->node[l+1][0][0][0].size(),0))));
	//this->qNode = this->eNode[l];
	this->qNode[l] = ite4(quantization,this->eNode[l]);
	}
      
      if(quantize){
	//this->qWeight = vector<vector<vector<vector<float>>>>
	//  (this->weight[wl].size(),vector<vector<vector<float>>>(this->weight[wl][0].size(),vector<vector<float>>(this->weight[wl][0][0].size(),vector<float>(this->weight[wl][0][0][0].size(),0))));
	
	//this->qWeight = this->weight[wl];
	this->qWeight[wl] = ite4(quantization,this->weight[wl]);
      }
      */
      
      fullyConnectedStep(l,wl,forward);
      wl += dir;
    }
    
    l += dir;
    //cout << l << "&" << endl;
    
    if(forward && l == layerN-1)
      break;
    
    else if(!forward && 0 >= l)
      break;
    
  }
  
  /*
  if(forward){
  for(int l = 0; l < layerN ; l++){
    cout << "layer " << l << endl;
    for(int i = 0; i < this->node[l][0][0].size() ; i++ ){
      for(int j = 0; j < this->node[l][0][0][0].size() ; j++ ){
	cout << this->node[l][0][0][i][j] << " ";
      }
      cout << "" << endl;
      cout << "" << endl;
    }
    cout << "" << endl;
  }
  }
  */
  /*
  if(!forward){
  for(int l = 0; l < layerN-1 ; l++){
    cout << "layerN" << l << endl;
    for(int i = 0; i < this->eNode[l][0][0].size() ; i++ ){
      for(int j = 0; j < this->eNode[l][0][0][0].size() ; j++ ){
	//if(this->eNode[l][0][0][i][j] > 0)
	//  cout << 1;
        //else if (this->eNode[l][0][0][i][j] == 0)
	cout << this->eNode[l][0][0][i][j] << " ";
	//else
	  //  cout << this->eNode[l][0][0][i][j] << " ";
      }
	
      cout << "" << endl;
    }
    cout << "" << endl;
  }
  }
  */
  /*
  for(int l = 0 ; l < weight.size() ; l++ ){
    cout << "layer"  << l << endl;
    for(int i = 0 ; i < weight[l].size() ; i++ )
      for(int o = 0 ; o < weight[l][i].size() ; o++ )
	for(int w = 0 ; w < weight[l][i][o].size() ; w++ )
	  for(int h = 0 ; h < weight[l][i][o][w].size() ; h++ ){
	    if(this->weight[l][i][o][w][h] > 0)
	      cout << this->weight[l][i][o][w][h] << " ";
            else if (this->weight[l][i][o][w][h] == 0)
	      cout << 0 << " ";
	    else
	      cout << this->weight[l][i][o][w][h] << " ";
	  }
     
  }
  */
}

void CNN::convolutionStep(int l,int wl,bool forward){
  
  //this copy let this program slower, but let implementation be simpler.
  //vector<vector<vector<vector<float>>>> vec;
  //vec = (forward) ? this->node : this->eNode;
  
  int inputFeatureN  = node[l][0].size();
  int outputFeatureN;
  int outputNodeW;
  int outputNodeH;
  int dir = (forward) ?  1 : -1;
  int cor = (forward) ?  1 :  0;
  
  outputFeatureN = node[l+cor][0].size();
  outputNodeW    = node[l+cor][0][0].size();
  outputNodeH    = node[l+cor][0][0][0].size();
  
  if(forward)
    this->node[l+dir] = ite4(zeroize,this->node[l+dir]);
  else
    this->eNode[l+dir] = ite4(zeroize,this->eNode[l+dir]);
  
  //vector<vector<float>> stock1(outputNodeW,vector<float>(outputNodeH,0));
  
  stock1 = vector<vector<float>> (outputNodeW,vector<float>(outputNodeH,0));
  
  for(int b = 0 ; b < this->batchSize ; b++ )
    for(int o = 0 ; o < outputFeatureN; o++ ){
      for(int i = 0 ; i < inputFeatureN ; i++ ){
	
	//convolution operation is coming....
	//I/O for convolution operation is going to be 2 dimension
	//input to the convolution ;  node  : this->node[l][b][i] // 2 dimension 
	//input to the convolution; weight : this->weight[l][i][o] // 2 dimension
	//output from convolution ; this->node[l+1][b][o][w][h] // 2 dimension

	stock1 = (forward) ? convolution(this->qNode[l][b][i],this->qWeight[wl][i][o])
	  : convolution(this->qeNode[l][b][i],this->qWeight[wl][o][i]);
	
	/*
	if(forward){
	  
	  stock1 = convolution(this->qNode[l][b][i],this->weight[wl][i][o]);
	
	} else
	  {
	    //if(quantize && quantizeN)
	    stock1 = convolution(this->qeNode[l][b][i],this->qWeight[wl][o][i]);
	    
	    //else
	    //stock1 = convolution(this->qeNode[l][b][i],this->weight[wl][o][i]);
	    
	  //stock1 = convolution(this->eNode[l][b][i],this->weight[wl][o][i]);
	  //cout << "hem" << endl;
	  //this->eNode[l+dir][b][o] utility::operator += convolution(this->eNode[l][b][i],this->weight[wl][o][i]);
	}
	*/
	//this->node[l+dir][b][o] = ite2(add,this->node[l+dir][b][o],stock1);
	
	for(int p = 0 ; p < stock1.size() ; p ++ )
	  for(int q = 0 ; q < stock1[p].size() ; q ++ ){
	    if(forward){
	      //	      cout << "stock1 " << stock1[p][q] << " ";
	      this->node[l+dir][b][o][p][q] += stock1[p][q];
	    }
	      
	    else
	      this->eNode[l+dir][b][o][p][q] += stock1[p][q];
	  }	
      }
    }
}

void CNN::poolingStep(int l,bool forward){
  
  int inputFeatureN  = this->node[l][0].size();
  int dir = (forward) ?  1 : -1;
  
  //int featureN = this->node[l+1][0].size();
  // this should be as half as previous layer assuming pooling size is 2.
  //int outputW = this->node[l+1][0][0].size();
  // this should be as half as previous layer assuming pooling size is 2.
  //int outputH = this->node[l+1][0][0][0].size();  
  //cout << "pooling operation! " << endl;

  if(forward)
    pooling(l,"max");
  
  for(int b = 0 ; b < this->batchSize ; b ++)
    //input and output for before or after of pooling should be same
    for(int i = 0 ; i < inputFeatureN ; i++ ){
      //pooling operation is coming....
      //I/O of pooling operation is going to be 2 dimension
      //input  ; node   : this->node[l][b][i]
      //input  ; weight : this->node[l][b][i]
      //output ; output : this->node[l+1][b][o]
      //assuming i == o
      //if(forward)
	//this->node[l+dir][b][i] = pooling(this->node[l][b][i],"max");
        //pooling(l,"max");
	//this->node[l+dir][b][i] = pooling(this->node[l][b][i],"max");
      if(!forward)
	this->eNode[l+dir][b][i] = rPooling(this->node[l][b][i], this->eNode[l][b][i],"max");
    }
  
}

void CNN::dimensionShift(int l,bool forward){
  
  //this is squashing process from 4 dimensional data to 2 dimension
  
  int inputFeatureN = this->node[l][0].size();
  int inputW        = this->node[l][0][0].size();
  int inputH        = this->node[l][0][0][0].size();
  int s = 0;
  
  for(int b = 0 ; b < this->batchSize ; b++ )
    for(int i = 0 ; i < inputFeatureN ; i++ )
      for(int w = 0; w < inputW ; w++)
	for(int h = 0; h < inputH; h++){
	  //just casting to 2D
	  if(forward){
	    //cout << "hei" << this->node[l][b][i][w][h] << " ";
	    this->node[l+1][0][0][b][s] = this->node[l][b][i][w][h];
	    //s ++;
	  } else {
	    this->eNode[l-1][b][i][w][h] = this->eNode[l][0][0][b][s];
	    //s ++;
	  }
	  s ++;
	}
}


void CNN::activationStep(int m,bool forward){
  
  //int inputFeatureN = this->node[m+dir][0].size();
  //int nodeW = this->node[m+dir][0][0].size();
  //int nodeH = this->node[m+dir][0][0][0].size();
  
  if(forward)
    this->node[m] = ite4(relu,this->node[m]);
  
  
}

void CNN::fullyConnectedStep(int l,int wl, bool forward){
  
  if(forward)
    this->node[l+1][0][0] = ite (sigmoid, dot(this->qNode[l][0][0],this->qWeight[wl][0][0]) );
  else
    this->eNode[l-1][0][0] = ite2 (deSigmoid, dot(this->qeNode[l][0][0],transpose(this->qWeight[wl][0][0])) , this->node[l][0][0] );
  
}

vector<vector<float>> CNN::rPooling(vector<vector<float>> data,vector<vector<float>> x,string type){
  
  int poolingSize = 2;
  int outputDataW = data.size();
  int outputDataH = data[0].size();
  float max = 0;
  pair<int,int> index;
  vector<float> buffer(4);
  vector<vector<float>> retArr(outputDataW,vector<float>(outputDataH,0));

  //cout << outputDataW << " : " << outputDataH << endl;
  
  for(int r = 0 ; r < outputDataW ; r+= 2)
    for(int c = 0 ; c < outputDataH ; c+= 2){
      
      buffer[0] = data[r][c];
      buffer[1] = data[r+1][c];
      buffer[2] = data[r][c+1];
      buffer[3] = data[r+1][c+1];
      max = 0;
      for(int i = 0 ; i < 4 ; i ++)
	if(buffer[i] > max){
	  max = buffer[i];
	  if(i == 0)
	    index = make_pair(r,c);
	  else if(i == 1 )
	    index = make_pair(r+1,c);
	  else if( i == 2 )
	    index = make_pair(r,c+1);
	  else
	    index = make_pair(r+1,c+1);
	}
      if(type == "max")
	retArr[index.first][index.second] = x[r/2][c/2];
      
      //if(type == "max")
      //retArr[index.first][index.second] = *std::max_element(buffer.begin(),buffer.end());
      //else if(type == "average"){
      //retArr[r][c] = (buffer[0] + buffer[1] + buffer[2] + buffer[3]) / (float)4.0;
      
    }
  
  return retArr;
}


void CNN::pooling(int l,string type){
  
  int poolingSize = 2;
  vector<float> buffer(4);
  float max = 0;
  for(int b = 0 ; b < node[l].size() ; b++)
    for(int i = 0 ; i < node[l][b].size() ; i++){

      for(int r = 0 ; r < node[l][b][i].size() ; r+= poolingSize)
	for(int c = 0 ; c < node[l][b][i][r].size() ; c+= poolingSize){
	  buffer[0] = node[l][b][i][r][c];
	  buffer[1] = node[l][b][i][r+1][c];
	  buffer[2] = node[l][b][i][r][c+1];
          buffer[3] = node[l][b][i][r+1][c+1];
	  
	  this->node[l+1][b][i][r/2][c/2] = (float)*std::max_element(buffer.begin(),buffer.end());
          //cout << "max" << max;
	  //this->node[l+1][b][i][r/2][c/2] = max;
	  //max = 0;
	  
	}
    }
}


/*
vector<vector<float>> CNN::pooling(vector<vector<float>> data,string type){
  
  int poolingSize = 2;
  //int outputDataW = this->node[l].size();
  //int outputDataW = this->node[l].size();
  //int outputDataW = this->node[l][][][][];
  
  int outputDataW = data.size() / poolingSize;
  int outputDataH = data[0].size() / poolingSize;
  
  //int buffer[4];
  vector<int> buffer(4);
  int average;
  float max = 0;
  int tt[4];
  int t1,t2,t3,t4;
  
  vector<vector<float>> retArr(outputDataW,vector<float>(outputDataH,0));
  
  for(int r = 0 ; r < data.size() ; r+= poolingSize)
    for(int c = 0 ; c < data[0].size() ; c+= poolingSize){
      cout << data[r][c] << " " << data[r+1][c] << endl;
      cout << data[r][c+1] << " " << data[r+1][c+1] << endl;
      t1 = data[r][c];
      t2 = data[r+1][c];
      t3 = data[r][c+1];
      t4 = data[r+1][c+1];
      if(type == "max"){
	if(t1 > t2){
	  max = t1;
	  if(max < t3){
	    max = t3;
	    if(max < t4)
	      max = t4;
	  }
	}
	
	retArr[r/2][c/2] = max;
	//cout << " max : " << max;
	max = 0;
	//retArr[r/2][c/2] = (float)*std::max_element(buffer.begin(),buffer.end());
      }
	
      else if(type == "average"){ 
	retArr[r][c] = (buffer[0] + buffer[1] + buffer[2] + buffer[3]) / (float)4.0;
      }
    }
  
  return retArr;  
}
*/


int main(int argc, char* argv[]){
  
  //plt::plot({1,2,3,4});
  ///plt::show();
  
  bool cifar10  = true;
  bool cifar100 = false;
  bool mnist = false;
  
  vector<vector<vector<vector<float>>>> inputImages;
  vector<vector<int>> teacher;
  
  if(cifar10){
    cout << "cifar-10 is going to be loaded " << endl;
    teacher = vector<vector<int>>(10000,vector<int>(10,0));
    inputImages = cifar10comeOn(teacher);
  }
  
  else if (cifar100){
    cout << "cifar-100 is going to be loaded " << endl;
    teacher = vector<vector<int>>(50000,vector<int>(100,0));
    inputImages = cifar100comeOn(teacher);  
  }
  
  
  else if(mnist){
    Mnist mni;
    cout << "MNIST is going to be loaded " << endl;
    inputImages = convert2Dto4D(mni.readTrainingFile("../data/mnist/train-images-idx3-ubyte"));
    teacher     = mni.readLabelFile("../data/mnist/train-labels-idx1-ubyte");
    cout << inputImages.size() << " : " << inputImages[0].size() << endl;
    cout << inputImages[0][0].size() << " : " << inputImages[0][0][0].size() << endl;
  }
  
  vector <pair<char,vector<int>>> model1 = {
    {'n',{1,28,28}},
    {'c',{3,28,28}},
    {'p',{3,14,14}},
    {'c',{5,14,14}},
    {'p',{5, 7, 7}},
    {'s',{5*7*7}},
    {'f',{200}},
    {'t',{10}}
  };
  
  vector <pair<char,vector<int>>> model2 = {
    {'n',{1,28,28}},
    {'c',{10,28,28}},
    {'p',{10,14,14}},
    {'s',{10*14*14}},
    {'f',{500}},
    {'t',{10}}
  };
  
  vector <pair<char,vector<int>>> model3 = {
    {'n',{3,32,32}},
    {'c',{5,32,32}},
    {'p',{5,16,16}},
    {'c',{8,16,16}},
    {'p',{8,8,8}},
    {'s',{8*8*8}},
    {'f',{200}},
    {'t',{100}}
  };
  
  vector<bool> nodeForwardB   = {true,true,true,false};
  vector<bool> nodeBackwardB  = {true,true,true,true};
  vector<bool> weightBinarize = {true,true,true,true};
  
  vector<vector<bool>> binarizeConfig;
  
  binarizeConfig.push_back(nodeForwardB);
  binarizeConfig.push_back(nodeBackwardB);
  binarizeConfig.push_back(weightBinarize);
  
  pair<int,int> kernelSize (3,3);
  int batchSize  = 1;
  float learningRate = 0.1;
  
  CNN cnn(inputImages,teacher,model3,binarizeConfig,kernelSize,batchSize,learningRate);
  
  cnn.mainLoop(10000);
  cnn.errorPlot();
  
  
  cout << "**************" << endl;
  
}

#include <iostream>
#include <vector>
#include <list>
//#include "bnn.h"
#include <math.h>
#include <random>
#include "matrix.cpp"
#include "mnistLoad.cpp"
#include <bitset>
#include "bitset2D.h"


random_device rd;
//mt19937 mt(rd());
uniform_real_distribution<float> dice(-1.0,1.0);
uniform_int_distribution<int> intDice(0,7);


class BCNN {
  
  vector<vector<vector<bitset<28>>>> inputData;
  
  vector<vector<vector<bitset<3>>>> w1;
  
  vector<vector<vector<bitset<28>>>> node1;
  
 public:
  BCNN (vector<vector<vector<bitset<28>>>> input) {
    inputData = input;
  }
  
  //vector<vector<vector<bitset<3>>>>
  void weightInit(){

    w1 = vector<vector<vector<bitset<3>>>>(1,vector<vector<bitset<3>>>(1,vector<bitset<3>>(3,bitset<3>())));
    
    //vector<vector<vector<bitset<3>>>> w;
    
    //cout << w1.size() << w1[0].size() << w1[0][0].size() << endl;
    
    for(int i = 0 ; i < w1.size() ; i++ )
      for(int j = 0 ; j < w1[i].size() ; j++ )
	for(int k = 0 ; k < w1[i][j].size() ; k++ ){
	  bitset<3> bs (intDice(rd));
	  w1[i][j][k] = (bs);
	}
    
    
    //cout << w1[0][0][0] << endl;
    //cout << w1[0][0][1] << endl;
    //return w1;
  }

  void mainLoop(){
    
    node1 = inputData;
    weightInit();
    bitMask();
    
  }
  
  void bitCount(){
    
    
  }
  
  void bitMask(){

    cout << w1[0][0][0] << endl;
    cout << node1[0][0][0] << endl;
    vector<vector<vector<bitset<28>>>> temp (3,vector<vector<bitset<28>>> (3,vector<bitset<28>>(28,bitset<28>())));
    
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++)
	temp[i][j] = node1[0][0];
    
    //for(int i = 0; i < 1 ; i++ ) //inputFeatureN
    //for(int j = 0; j < 1 ; j++) // outputFeatureN
    for(int r = 0 ; r < 3 ; r ++ ){ //dataDimensionRow
	  //bitset<28> temp;
      for(int c = 0 ; c < 3 ; c ++ ){ //weightHeight
	    //cout << w1[i][j][r][c] << endl;
	    //if (w1[i][j][r][c] == 0){
	    //}	
	    //temp.set();
	    //cout << node1[0][i][c] << endl;
	if (w1[0][0][r][c] == 0){
	  for(int i = 0; i < 28; i++)
	    //for(int j = 0; j < 28; j++)
	    temp[r][c][i].reset();	  
	}
	for(int i = 0; i < 28; i++)
	  cout << temp[r][c][i] << endl;
      }
    }
    
    /*
    bitset<8> bs = 0b01110111;
    cout << bs << endl;
    bs >>= 1;
    //bs <<= 1;
    cout << bs << endl;
    
    for (int r = 0 ; r < 3 ; r++)
      for (int c = 0 ; c < 3 ; c++ ){
	
	if(r == 0)
	  for(int i = 0 ; i < 28 ; i++ )
	    temp[r][c][i] <<= 1;
	
	else if(r == 2)
	  for(int i = 0 ; i < 28 ; i++ )
	    temp[r][c][i] >>= 1;
	
	
	if( c == 0 )
	  temp[r][c] <<= 1;
	else if( c == 2 )
	  temp[r][c] >>= 1;
	*/
     
    
    vector<vector<bitset<9>>> temp2(28,vector<bitset<9>>(28,bitset<9>()));
    
    for (int r = 0 ; r < 3 ; r++)
      for (int c = 0 ; c < 3 ; c++ )
	for (int rr = 1 ; rr < 27 ; rr++){
	  for (int cc = 1 ; cc < 27 ; cc++ ){
	    if(r == 0 && c == 0)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr+1][cc+1];
	    if(r == 1 && c == 0)
		temp2[rr][cc][r*3+c] = temp[r][c][rr][cc+1];
	    if(r == 2 && c == 0 && rr != 0 && cc != 27)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr-1][cc+1];
	    if(r == 0 && c == 1 && rr != 0 && cc != 27)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr+1][cc];
	    if(r == 1 && c == 1 && rr != 0 && cc != 27)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr][cc];
	    if(r == 2 && c == 1)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr-1][cc];
	    if(r == 0 && c == 2)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr+1][cc-1];
	    if(r == 1 && c == 2)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr][cc-1];
	    if(r == 2 && c == 2)
	      temp2[rr][cc][r*3+c] = temp[r][c][rr-1][cc-1];
	  }
	}
    
    for(int i = 0; i < 28; i++){
      for(int j = 0; j < 28; j++)
	cout << temp2[i][j].count();
      cout << "" << endl;
    }
    
    //cout << temp2[0][0] << endl;
    //.erase(remove(temp.begin(), temp.end(), 0), temp.end());
    
  }
  
  
};

vector<vector<vector<vector<float> > > > convert2Dto4D (vector<vector<float>> x){
  
  int dataN = x.size();
  int dataD = x[0].size();
  
  vector<vector<vector<vector<float> > > > vecR(60000,vector<vector<vector<float> > >(1,vector<vector<float> > (28,vector<float>(28))));
  
  for (int i = 0 ; i < dataN ; i++ )
    for(int j = 0 ; j < dataD ; j++ ){
      int r = j / 28;
      int c = j % 28;
      vecR[i][0][r][c] = x[i][j];
    }
  
  return vecR;
  
}

vector<vector<vector<bitset<28>>>> inputBinarize(vector<vector<vector<vector<float>>>> x){
  
  vector<vector<vector<bitset<28>>>> s (60000,vector<vector<bitset<28>>>(1,vector<bitset<28>>(28,bitset<28>())));
  
  for(int k = 0 ; k < x.size() ; k ++)
    for(int f = 0 ; f < x[0].size() ; f++ ){
      for(int i = 0 ; i < 28 ; i++){
	bitset<28> bs;
        for(int j=0 ; j < 28 ; j++ ){
          if(x[k][f][i][27-j] > 0.5)
	    bs[j] = 1;
          else
	    bs[j] = 0;
          }
	s[k][f][i] = bs;
      }
    }
  
  return s;
  
}




int main(int argc,char* argv[]){
  
  //dunsigned elm : 1 << 4;
  
  //unsigned char *arr;
  //unsigned char a = 0b00110011;
  //cout << static_cast<bitset<8>> (a) << endl;

  for( int i = 0 ; i < 10 ; i++ )
    cout << intDice(rd) << endl;
  
  
  Mnist mnist;
  
  vector<vector<float>> mnistImages;
  vector<vector<vector<vector<float>>>> mnistImages4D;
  vector<vector<int>> mnistTeacher;
  
  mnistImages  = mnist.readTrainingFile("/home/hiroshi/Downloads/mnist/train-images-idx3-ubyte");
  mnistTeacher = mnist.readLabelFile("/home/hiroshi/Downloads/mnist/train-labels-idx1-ubyte");
  
  bitset<8> b = (0b01101111);
  
  mnistImages4D = convert2Dto4D(mnistImages);
  
  cout << mnistImages4D.size() << " , " << mnistImages4D[0].size() << " , " << mnistImages4D[0][0].size() << endl;
  
  
  vector<vector<vector<bitset<28>>>> inputBitImage;
  inputBitImage = inputBinarize(mnistImages4D);
  
  BCNN bcnn(inputBitImage);

  bcnn.mainLoop();
  
  
}



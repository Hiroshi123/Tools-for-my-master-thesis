#include <iostream>
#include <vector>
#include <bitset>
#include <boost/dynamic_bitset.hpp>


using namespace std;

char strToChar(const char* str) {
  
  char parsed = 0;
  
  for (int i = 0; i < 8; i++) {
    //cout << str[i] ;
    if (str[i] == '1') {
      parsed |= 1 << (7 - i);
    }
  }
  return parsed;
}

template<typename T>
vector<vector<char>> inputBinarize(vector<vector<T>> x){
  
  int u  = x[0].size() / 8 ;
  if(x[0].size() % 8 != 0)
    u ++;
  
  vector<vector<char>> y;
  for(int d = 0 ; d < x.size(); d++ ){
    vector<char> a;
    for(int i = 0 ; i < u ; i ++ ){
      char t[8];
      for(int j = 0 ; j < 8 ; j++ ){
	if(i * 8 + j >= x[d].size() )
	  t[j] = '0';
	else
	  t[j] = (x[d][i * 8 + j] > 0.5) ? '1' : '0';
      }
      a.push_back(strToChar(t));
      //printBinary(strToChar(tempChar));
    }
    y.push_back(a);
  }
  return y;
}

vector<vector<float>> bitDot(vector<vector<float>> n,vector<vector<float>> w,bool forward){
  
  //declare return matrix here
  vector<vector<float>> ret(n.size(),vector<float>(w[0].size(),0));
  
  const int t = w.size();
  
  boost::dynamic_bitset<>   nn(t);
  boost::dynamic_bitset<>   ww(t);
  boost::dynamic_bitset<> temp(t);
  
  //if you want to set normal bitset, the length of bitset have to be fed directly without being put on any variables.
  //bitset<1000>nn;
  //bitset<1000> ww;
  //bitset<1000>temp;
  
  //std::cout << n[0][0] << std::endl;
  
  int bitCount1,bitCount2;
  //bitset setting
  for(int o = 0 ; o < w[0].size() ; o++ ){
    //there might be better way to feed into binary data into dynamic bitset than following
    for(int i = 0; i < w.size(); i ++){
      
      if(o == 0 && n[0][i] == 1)
	nn.set(i);
      if(w[i][o] == 1)
	ww.set(i);
    }
    
    if(forward){
      //as ww is gonna be converted after following command original value of weight have to be evacuated into different memory.
      
      temp = ww;
      ww &= nn;
      bitCount1 = ww.count();
      temp.flip(); //temp = ~temp; 
      temp &= nn;
      bitCount2 = temp.count();
      //so far only for batch operation
      ret[0][o] = (float)(bitCount1 - bitCount2);
    } else {
      //it means following is backward pass!
      ww ^= nn; //xor
      int te = ww.count();
      // (size - count) - bitcount = size - 2 * bitcount()
      ret[0][o] = (float) (t - 2 * te);
      //cout << (t - 2 * te) << endl;
    }
    
    ww.reset();
    temp.reset();
    
  }
  
  //std::cout << " kk "<< ret[0][0] << std::endl;
  
  return ret;
}


//slow version..
/*
vector<vector<float>> bitForward(vector<vector<float>> node,vector<vector<float>> w){
  
  vector<vector<float>> ret(node.size(),vector<float>(w[0].size(),0));
  vector<vector<char>> s = inputBinarize(transpose(w));
  vector<vector<char>> n = inputBinarize(node);
  
  bitset<8> tempBit1;
  bitset<8> tempBit2;
  char bitSave[8];
  int outputNodeN =   s.size();
  int inputNodeN  = s[0].size();
  int bitStock1,bitStock2;
  for(int j = 0 ; j < outputNodeN ; j ++ ){
    bitStock1 = 0;
    bitStock2 = 0;
    for(int i = 0 ; i < inputNodeN ; i ++ ){
      tempBit1 = (s[j][i] & n[0][i]);  //bit And operation
      tempBit2 = ((~s[j][i]) & n[0][i]);  //bit And operation after flipping (to simulate negative weight)
      bitStock1 += tempBit1.count();
      bitStock2 += tempBit2.count();
    }
    
    //cout << bitStock1 << " : " << bitStock2 << endl;
    //bitSave[j%8] = bitStock1 > bitStock2 ? '1' : '0';
    //cout << bitSave[j%8] << endl;
    //if(j % 8 == 7 || j == s.size()-1)
    //  ret[0][j/8] = strToChar(bitSave);
    
    ret[0][j] = (float)(bitStock1 - bitStock2);
    
  }
  
  return ret;
}
*/

auto quants = [](auto x){return x > 0 ? 1 : 0;};

//auto quantization = [](float x) { return x > 0 ? 1 : -1 ;};

float qf1(float x,float a){
  int th = 7;
  //std::cout << x << std::endl;
  if(x > th)
    return 1.0;
  else if(x < -th)
    return -1.0;
  else
    return 0;
}

float quantization(float x) { return x > 0 ? 1 : -1 ;}

float quantization2(float x,float a) { return a > 0 && x > 0 ? 1 : -1 ;}

float quantization2E(float x,float a) { return x > 0 ? 1 : -1 ;}
float quantization2Ef(float x,float a) { return x > 0 ? 0.0 : -0.0 ;}

float quantizationN(float x) { return x > 0.5 ? 1 : 0 ;}
float quantizationNS(float x) { return x > 0.5 ? 1 : -1 ;}
float quantizationN2(float x,float a) { return a > 0 && x > 0 ? 1 : 0 ;}

float ternalization(float x) {
  if(x > 0.25)
    return 1;
  else if(x < -0.25)
    return -1;
  else
    return 0;
}


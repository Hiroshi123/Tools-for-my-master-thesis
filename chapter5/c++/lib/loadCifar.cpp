#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

vector<vector<vector<vector<float>>>>
read_batch(string filename,vector<vector<int>> &teacher,bool cifar100){

  vector<vector<vector<vector<float>>>> retVec;
  
  if(cifar100){
    retVec = vector<vector<vector<vector<float>>>>
      (50000,vector<vector<vector<float>>>(3,vector<vector<float>>(32,vector<float>(32,0))));
    //error aquisition
    if(teacher.size() != 50000){
      cerr << "number of input image have to be 10000" << endl;
      assert(teacher.size() == 50000);
    }
    if(teacher[0].size() != 100){
      cerr << "number of output class have to be 100" << endl;
      assert(teacher[0].size() == 100);
    }
    
  }
  
  else{
    retVec = vector<vector<vector<vector<float>>>>
      (10000,vector<vector<vector<float>>>(3,vector<vector<float>>(32,vector<float>(32,0))));
    
    //error aquisition
    if(teacher.size() != 10000){
      cerr << "number of input image have to be 10000" << endl;
      assert(teacher.size() == 10000);
    }
    if(teacher[0].size() != 10){
      cerr << "number of output class have to be 10" << endl;
      assert(teacher[0].size() == 10);
    }
  }
  
  ifstream file (filename, ios::binary);
  if (file.is_open())
    {
      int number_of_images = 10000;
      int n_rows = 32;
      int n_cols = 32;
      for(int i = 0; i < number_of_images; ++i)
	{
	  unsigned char tplabel = 0;
	  file.read((char*) &tplabel, sizeof(tplabel));
	  if(cifar100){
	    unsigned char sublabel = 0;
	    file.read((char*) &sublabel, sizeof(sublabel));
	    //if you want to store sub label for cifar 100, store following sub label
	    
	  }
	  
	  //vector<Mat> channels;
	  //Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
	  for(int ch = 0; ch < 3; ++ch){
	    //Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
	    for(int r = 0; r < n_rows; ++r){
	      for(int c = 0; c < n_cols; ++c){
		unsigned char temp = 0;
		file.read((char*) &temp, sizeof(temp));
		retVec[i][ch][r][c] = (float) temp / (float) 255;
	      }
	    }
	  }
	  teacher[i][int(tplabel)] = 1;
	}
    }
  
  return retVec;
  
}

vector<vector<vector<vector<float>>>> cifar10comeOn(vector<vector<int>> teacher){
  
  string filename[5];
  
  filename[0] = "../../data/cifar/cifar10/data_batch_1.bin";
  filename[1] = "../../data/cifar/cifar10/data_batch_2.bin";
  filename[2] = "../../data/cifar/cifar10/data_batch_3.bin";
  filename[3] = "../../data/cifar/cifar10/data_batch_4.bin";
  filename[4] = "../../data/cifar/cifar10/data_batch_5.bin";
  
  string test;
  test = "../../data/cifar/cifar10/cifar-10-batches-bin/test_batch.bin";
  vector<vector<vector<vector<float>>>> inputImages;
  
  //vector<vector<int>> teacher
  //  (10000,vector<int>(10,0));
  
  inputImages = read_batch(filename[0],teacher,false);
  
  //**************************************************//
  //image description
  cout << inputImages[0][0][0][0] << endl;
  cout << inputImages.size() << ", " << inputImages[0].size() << ", "
       << inputImages[0][0].size() << ", " << inputImages[0][0][0].size() << endl;
  
  return inputImages;  
}

vector<vector<vector<vector<float>>>> cifar100comeOn(vector<vector<int>> teacher){
  
  string train;
  train = "../../data/cifar/cifar100/train.bin";

  string test;
  test = "../../data/cifar/cifar100/test.bin";

  vector<vector<vector<vector<float>>>> inputImages;
  
  //vector<vector<int>> teacher
  //  (10000,vector<int>(10,0));

  bool cifar100 = true;
  
  inputImages = read_batch(train,teacher,cifar100);
  
  //**************************************************//
  //image description
  cout << inputImages[0][0][0][0] << endl;
  cout << inputImages.size() << ", " << inputImages[0].size() << ", "
       << inputImages[0][0].size() << ", " << inputImages[0][0][0].size() << endl;
  
  return inputImages;  
}


/*
int main(){
  
  string filename[5];
  
  filename[0] = "../../data/cifar/data_batch_1.bin";
  filename[1] = "../../data/cifar/data_batch_2.bin";
  filename[2] = "../../data/cifar/data_batch_3.bin";
  filename[3] = "../../data/cifar/data_batch_4.bin";
  filename[4] = "../../data/cifar/data_batch_5.bin";
  
  string test;
  test = "cifar-10-batches-bin/test_batch.bin";
  
  vector<vector<vector<vector<float>>>> inputImages;
  vector<vector<float>> teacher
    (10000,vector<float>(10,0));
  
  inputImages = read_batch(filename[0],teacher);

  for(int i = 0; i < 10; i++)
    cout << teacher[1][i];
  
  cout << inputImages[0][0][0][0] << endl;
  cout << inputImages.size() << ", " << inputImages[0].size() << ", "
       << inputImages[0][0].size() << ", " << inputImages[0][0][0].size() << endl;
  
}
*/
  



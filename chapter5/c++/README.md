# binary neural network

Binary neural network implementation in c++.

Dataset for benchmark measurement is Mnist.


## Execution

To compile, run with c++11 std.

'g++ -std=c++11 mlp.cpp '

### mlp.cpp

Refererence implementation on chainer (https://github.com/hillbig/binary_net)

forward pass (-f)
x : non-quantization
b : quantization to 1 or 0


backward pass (-e)

x : non-quantization
b : quantization to 1 or -1
t : quantization to 1, 0, or -1


weight (-f)

x : non-quantization
b : quantization to 1 or -1
t : quantization to 1, 0, or -1


example

#### standard multi layer perceptron

`./a.out -f x -e x -w x` 

#### multi layer perceptron with binary quantized weight

`./a.out -f x -e x -w b` 

#### multi layer perceptron with binary quantized weight and node but only in forward pass

`./a.out -f b -e x -w b`

#### multi layer perceptron with binary quantized nodes & weight

`./a.out -f b -e b -w b`


Note this is not convolution neural network.

## Description

There are two ways to compute to calculate feed-forward and feed-back operation.

One is just another dot operation with a couple of matrixes.

Another is bit calculation which consists of set of procedure ; mutiple basic bit operation.

It can be switched with internal setting, specifically setting Bool bitCalc. 

Feed-forward and Feed-back in bit calculation is not identical assuming activation function is either sigmoid or rectified linear unit.

This is because these activation function squash the range of values which nodes are able to hold, namely all of values in forward process is set to be positive with these. 

On backward pass, error propagated value can be either positive or negative so weights could be updated in both direction.

Procedure of bit calculation is written as "bitDot" method on a file named "bnnSub.cpp".

So far, it is not clear how much this operation could contribute its acceration but should be performative for convolution operation.

### bnn.cpp

Bit calculation based operation with no floating value at all.

This is out of any papers which comes from my curiocity.

Model size is going to be by far tiny rather than original one.

Note this is still on the way...

### bcnn.cpp

Binary convolutional neural network on the XNorNet paper.

This is also on the way...




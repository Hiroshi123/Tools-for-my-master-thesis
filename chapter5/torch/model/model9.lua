
--As this model is for cifar10, input picture is resized as 3 * 32 * 32

model = {}

function setModel()
   
   model = {
      
      operations = {
	 
	 "setData3D",
	 "convolution",
	 "relu",
	 "maxPooling",
	 "convolution",
	 "relu",
	 "maxPooling",
	 "convolution",
	 "relu",
	 "maxPooling",
	 "convolution",
	 "relu",
	 "maxPooling",
	 "convolution",
	 "relu",
	 --"maxPooling",
	 "reshape",
	 "linear",
	 "sigmoid"
      },
      
      --following is a bit to represent each activations of nodes and weights
      
      bitForward = {
	 32,32,32,32,32,32
      },
      
      bitBackward = {
	 32,32,32,32,32,32
      },
      
      bitWeight = {
	 32,32,32,32,32,32
      },
      
      bitUpdate = {
	 
      },
      
      len = 18,
      nodeLen   = 12,
      weightLen = 6,
      
      
      kernel = {
	 width  = 3,
	 height = 3
      },
      
      layerN = {
	 
	 {3,32,32},--setdata
	 {5,32,32},--convolution
	 {5,32,32},--relu
	 {5,16,16},--pooling
	 {20,16,16},--convolution
	 {20,16,16},--relu
	 {20, 8, 8},--pooling
	 {50, 8, 8},--convolution
	 {50, 8, 8},--relu
	 {50, 4, 4},--pooling
	 {30, 4, 4},--convolution
	 {30, 4, 4},--relu
	 {30, 2, 2},--pooling
	 {10, 2, 2},--convolution
	 {10, 2, 2},--relu
	 --{10, 1, 1},--pooling
	 10*2*2,    --reshape
	 10,        --linear
	 10         --sigmoid
      },
      
      
      node    = {},
      nodeB   = {},
      nodeT   = {},
      eNode   = {},
      eNodeB  = {},
      weight  = {},
      weightB = {},
      dWeight = {},
      teacher = {}
   }
   
end


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
	 "reshape",
	 "linear",
	 "sigmoid",
	 "linear",
	 "sigmoid"
      },
      
      --following is a bit to represent each activations of nodes and weights
      
      bitForward = {
	 32,32,32,32
      },
      
      bitBackward = {
	 32,32,32,32
      },
      
      bitWeight = {
	 32,32,32,32
      },
      
      bitUpdate = {	 
      },
      
      len = 12,
      nodeLen   = 8,
      weightLen = 4,
      
      kernel = {
	 width  = 3,
	 height = 3
      },
      
      layerN = {
	 
	 {3,32,32},--setdata
	 {10,32,32},--convolution
	 {10,32,32},--relu
	 {10,16,16},--pooling
	 {20,16,16},--convolution
	 {20,16,16},--relu
	 {20, 8, 8},--pooling
	 20*8*8,    --reshape
	 300,      --linear
	 300,      --sigmoid
	 10,       --linear
	 10        --sigmoid
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

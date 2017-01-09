
--As this model is for cifar10, input picture is resized as 3 * 32 * 32

model = {}


function setModel()
   
   model = {
      
      operations = {
	 "setData",
	 "linear",
	 "sigmoid",
	 "linear",
	 "sigmoid"
      },
      
      --following is a bit to represent each activations of nodes and weights
      
      bitForward = {
	 32,32
      },
      
      bitBackward = {
	 32,32
      },
      
      bitWeight = {
	 32,32
      },
      
      bitUpdate = {
	 
      },
      
      len = 5,
      nodeLen   = 3,
      weightLen = 2,
      
      layerN = {
	 3*32*32,1000,1000,10,10
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


model   = {}

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
	 1,1
      },
      
      bitBackward = {
	 1,1
      },
      
      bitWeight = {
	 1,1
      },
      
      bitUpdate = {
	 
      },
      
      len = 5,
      nodeLen   = 3,
      weightLen = 2,
      
      layerN = {
	 28*28,500,500,10,10
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

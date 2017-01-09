
model   = {}

function setModel() 
   --local params1 = dataset.trainInput:size()
   
   --params = {
   --   dataN   = params1[1],
   --   inputN  = params1[2] * params1[3],
   --   hiddenN = 500,
   --   outputN = 10
   --}
   
   model = {
      
      operations = {
	 "setData",
	 "linear",
	 "sigmoid",
	 "linear",
	 "sigmoid",
	 "linear",
	 "sigmoid"
      },
      
      bitForward = {
	 1,1,1
      },
      
      bitBackward = {
	 32,32,32
      },
      
      bitWeight = {
	 1,1,32
      },
      
      bitUpdate = {
	 
      },
      
      len = 7,
      nodeLen   = 4,
      weightLen = 3,
      
      layerN = {
	 28*28,500,500,300,300,10,10
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

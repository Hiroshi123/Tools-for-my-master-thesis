

function feedForward(i,train_or_test)
   
   local nodeIndex   = 1
   local weightIndex = 1
   
   poolingFuncTable = {}

   --print(i,"-----")
   if(i == 6703) then
      print("-----------------------")
   end
   
   for k = 1,model.len do
      if ( model.operations[k] == "setData" or model.operations[k] == "setData3D" ) then
	 for b = 1,opt.bs do
	    model.node[1][b]:fill(0)
	    if(train_or_test) then
	       model.node[1][b] = dataset.trainInput[i+b-1]
	    else
	       model.node[1][b] = dataset.testInput[i+b-1]
	    end
	 end	 

	 --nodeIndex = nodeIndex + 1
      end
      
      if (model.operations[k] == "convolution") then
	 
	 model.node[nodeIndex+1]:fill(0)
	 
	 if (model.bitForward[weightIndex] == 32) then
	    model.nodeB[nodeIndex] = model.node[nodeIndex]:clone()
	 else
	    local t1 = torch.pow(2,model.bitForward[weightIndex])
	    model.nodeB[nodeIndex] = torch.floor(model.node[nodeIndex] * t1) / t1
	 end
	 
	 if (model.bitWeight[weightIndex] == 32) then
	    model.weightB[weightIndex] = model.weight[weightIndex]:clone()
	 else
	    local t2 = torch.pow(2,model.bitWeight[weightIndex])
	    model.weightB[weightIndex] = torch.floor((model.weight[weightIndex] + 1) * ( t2 / 2 ) ) * (4 / t2) - 1
	 end
	 
	 if(opt.GPU >= 0) then
	    local seq = nn.Sequential()
	    local conv = cudnn.SpatialConvolution(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height,1,1,1,1):cuda()	    
	    conv.weight = model.weightB[weightIndex]:transpose(1,2)
	    model.node[nodeIndex+1] = conv:forward(model.nodeB[nodeIndex])
	    
	 else
	    
	    local conv = nn.SpatialConvolution(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height,1,1,1,1)
	    --print(model.weight[weightIndex]:size())
	    --conv.weight = model.weight[weightIndex]:transpose(1,2)
	    conv.weight = model.weightB[weightIndex]:transpose(1,2)
	    --model.node[nodeIndex+1] = conv:forward(model.node[nodeIndex])
	    model.node[nodeIndex+1] = conv:forward(model.nodeB[nodeIndex])
	 end 
	 nodeIndex   = nodeIndex + 1
	 weightIndex = weightIndex + 1
	 
      end
      
      if (model.operations[k] == "maxPooling") then
	 local resW = model.node[nodeIndex]:size()[3]
	 local resH = model.node[nodeIndex]:size()[4]
	 --if (opt.GPU >= 0) then 
	 if (opt.GPU >= 0) then
	    local pooling1  = cudnn.SpatialMaxPooling(2,2):cuda()
	    table.insert(poolingFuncTable,pooling1)

	    model.node[nodeIndex+1] = pooling1:forward(model.node[nodeIndex])
	    
	 else

	    pooling1 = nn.SpatialMaxPooling(2,2)-- :cuda()

	    table.insert(poolingFuncTable,pooling1)
	    model.node[nodeIndex+1] = pooling1:forward(model.node[nodeIndex])
	    
	 end

	 nodeIndex = nodeIndex + 1
	 
      end
      
      if (model.operations[k] == "reshape") then
	 model.node[nodeIndex+1][opt.bs] =
	    torch.reshape(model.node[nodeIndex][opt.bs],
			  model.node[nodeIndex][1]:size()[1] *
			     model.node[nodeIndex][1]:size()[2] *
			     model.node[nodeIndex][1]:size()[3])
	 nodeIndex = nodeIndex + 1
	 
      end
      
      if (model.operations[k] == "linear") then
	 
	 model.node[nodeIndex+1]:fill(0)
	 
	 --binarize
	 --
	 --model.weightB[nodeIndex] = model.weight[nodeIndex]:clone()
	 
	 if (model.bitForward[weightIndex] == 32) then
	    model.nodeB[nodeIndex] = model.node[nodeIndex]:clone()
	 else
	    local t1 = torch.pow(2,model.bitForward[weightIndex])
	    model.nodeB[nodeIndex] = torch.floor(model.node[nodeIndex] * t1) / t1
	 end
	 
	 if (model.bitWeight[weightIndex] == 32) then
	    model.weightB[weightIndex] = model.weight[weightIndex]:clone()
	 else
	    local t2 = torch.pow(2,model.bitWeight[weightIndex])
	    model.weightB[weightIndex] = torch.floor((model.weight[weightIndex] + 1) * ( t2 / 2 ) ) * (4 / t2) - 1
	 end
	 
	 --model.weightB[nodeIndex] = floor_ceil(model.weight[nodeIndex] * t2) / t2
	 --model.weightB[nodeIndex] : apply(floor_ceil)
	 --model.weightB[nodeIndex] = model.weightB[nodeIndex] / t2
	 
	 --print(model.weightB[nodeIndex])
	 --print(model.node[nodeIndex])
	 --if (model.bitForward[nodeIndex] == 1) then
	 --   model.nodeB[nodeIndex]
	    --model.nodeB[nodeIndex]:apply(binarizeP)
	 --end	 

	 --if (model.bitForward[nodeIndex] == 2) then
	 --   model.nodeB[nodeIndex]:apply(binarizeP2)
	 --end 
         
	 --if (model.bitWeight[nodeIndex] == 1) then
	 --   model.weightB[nodeIndex]:apply(binarize)
	 --end
	 --]]
	 
	 --model.node[nodeIndex+1]:addmm(model.node[nodeIndex],model.weight[weightIndex])	 
	 
	 local inputF  = model.weight[weightIndex]:size()[1]
	 local outputF = model.weight[weightIndex]:size()[2]
	 
	 if(opt.GPU >= 0) then 
	    
	    --local dot = nn.Linear(inputF,outputF):cuda()
	    --dot:forward(model.nodeB[nodeIndex],model.weightB[weightIndex])
	    model.node[nodeIndex+1]:addmm(model.nodeB[nodeIndex],model.weightB[weightIndex]):cuda()
	    
	 else
	    model.node[nodeIndex+1]:addmm(model.nodeB[nodeIndex],model.weightB[weightIndex])
	    --nn.Linear(inputF,outputF):forward(model.nodeB[nodeIndex],model.weightB[weightIndex])
	 end
	 
	 --model.node[nodeIndex+1]:addmm(model.nodeB[nodeIndex],model.weightB[weightIndex])
	 
	 nodeIndex   = nodeIndex + 1
	 weightIndex = weightIndex + 1	 
	 
      end
      
      if (model.operations[k] == "relu") then
	 if ( opt.GPU >= 0 ) then
	    local relu = cudnn.ReLU():cuda()
	    model.node [nodeIndex] = relu:forward(model.node[nodeIndex])
	 else 
	    model.node [nodeIndex] = nn.ReLU():forward(model.node[nodeIndex])
	 end
	 
	 model.nodeT[nodeIndex-1] = model.node[nodeIndex]:clone()
	 
	 --model.nodeT[nodeIndex-1]:apply(deSigmoid)
	 
      end
      
      if (model.operations[k] == "sigmoid") then
	 if(opt.GPU >= 0) then
	    local sigmoid = cudnn.Sigmoid():cuda()
	    model.node[nodeIndex] = sigmoid:forward(model.node[nodeIndex])

	 else
	    model.node[nodeIndex] = nn.Sigmoid():forward(model.node[nodeIndex])
	 end      
	 
	 model.nodeT[nodeIndex-1] = model.node[nodeIndex]:clone()
	 model.nodeT[nodeIndex-1]:apply(deSigmoid)
      end
      
   end
   
end



function feedBack(i)
   
   --this have to be -1 because eNode is node - 1
   local m = model.nodeLen - 1
   local wIndex = model.weightLen
   local poolingIndex = poolingMaxIndex
   
   for k = model.len-1,3,-1 do
      
      if (model.operations[k] == "convolution") then
	 
	 model.eNode[m-1]:fill(0)
	 
	 if (model.bitBackward[wIndex] == 32) then
	    model.eNodeB[m] = model.eNode[m]:clone()
	 else
	    local t3 = torch.pow(2,model.bitBackward[wIndex])
	    model.eNodeB[m] = torch.floor((model.eNode[m] + 1) * (t3 / 2 )) * ( 4 / t3 ) - 1
	 end	 
	 
	 if (opt.GPU >= 0) then
	    
	    local conv2  = cudnn.SpatialConvolution(model.layerN[k][1],model.layerN[k-1][1],model.kernel.width,model.kernel.height,1,1,1,1):cuda()
	    --conv2.weight = model.weight[wIndex]--:transpose(1,2)
	    --model.eNode[m-1] = conv2:forward (model.eNode[m])
	    conv2.weight = model.weightB[wIndex]--:transpose(1,2)
	    model.eNode[m-1] = conv2:forward (model.eNodeB[m])
	    
	 else 
	    
	    local conv2  = nn.SpatialConvolution(model.layerN[k][1],model.layerN[k-1][1],model.kernel.width,model.kernel.height,1,1,1,1)
	 
	    --conv2.weight = model.weight[wIndex]--:transpose(1,2)
	    --model.eNode[m-1] = conv2:forward (model.eNode[m])
	    conv2.weight = model.weightB[wIndex]--:transpose(1,2)
	    model.eNode[m-1] = conv2:forward (model.eNodeB[m])

	 end

	 m = m - 1
	 wIndex = wIndex - 1
	 
      end
      
      if (model.operations[k] == "maxPooling") then
	 model.eNode[m-1]:fill(0)
	 if (opt.GPU >= 0) then
	    --print(poolingFuncTable[poolingIndex]:updateOutput)
	    model.eNode[m-1] = poolingFuncTable[poolingIndex]:updateGradInput(model.node[m],model.eNode[m]):cuda()
	    --print("------")
	    --print(model.eNode[m]:size())
	    --print(model.eNode[m-1]:size())
	    --model.eNode[m-1] = unPooling1:updateOutput(model.eNode[m])
	    
	 else
	    unPooling1 = nn.SpatialMaxUnpooling(poolingFuncTable[poolingIndex]) --(pooling1)
	    model.eNode[m-1] = unPooling1:updateOutput(model.eNode[m])
	    
	 end
	 
	 m = m - 1
	 poolingIndex = poolingIndex - 1
	 
      end
      
      if (model.operations[k] == "reshape") then
	 model.eNode[m-1]:fill(0)
	 model.eNode[m-1][opt.bs] = torch.reshape(model.eNode[m][opt.bs],
						 model.eNode[m-1][1]:size()[1] ,
						 model.eNode[m-1][1]:size()[2] ,
						 model.eNode[m-1][1]:size()[3])
	 
	 m = m - 1
	 
      end
      
      if (model.operations[k] == "sigmoid" ) then
	 
	 model.eNode[m]:cmul(model.nodeT[m])
	 
	 --m = m - 1
	 
      end
      
      if (model.operations[k] == "linear") then
	 -- model.weightB[m] --= model.weight[m]:apply(binarize)
	 -- model.eNodeB[m] = model.eNode[m]:clone()
	 
	 model.eNode[m-1]:fill(0)
	 
	 if (model.bitBackward[wIndex] == 32) then
	    model.eNodeB[m] = model.eNode[m]:clone()
	 else
	    local t3 = torch.pow(2,model.bitBackward[wIndex])
	    model.eNodeB[m] = torch.floor((model.eNode[m] + 1) * (t3 / 2 )) * ( 4 / t3 ) - 1
	 end
	 
	 --if(model.bitBackward[m] == 1) then
	 --   model.eNodeB[m]:apply(binarize)
	 --end
	 --if (model.bitBackward[m] == 2) then
	 --   model.eNodeB[m]:apply(binarize2)
	 --end
	 
	 --print(model.weightB[wIndex]:size())
	 --print(model.eNodeB[m]:size())
	 
	 --model.eNode[m-1]:addmm(model.eNode[m],model.weight[wIndex]:transpose(1,2))
	 
	 local inputF  = model.weight[wIndex]:size()[2]
	 local outputF = model.weight[wIndex]:size()[1]
	 
	 if(opt.GPU >= 0) then
	    
	    --local dot = nn.Linear(inputF,outputF):cuda()
	    --dot:forward(model.eNodeB[m],model.weightB[wIndex]:transpose(1,2))
	    model.eNode[m-1]:addmm(model.eNodeB[m],model.weightB[wIndex]:transpose(1,2)):cuda()
	    
	 else
	    
	    --nn.Linear(model.eNodeB[m],model.weightB[wIndex]:transpose(1,2))
	    model.eNode[m-1]:addmm(model.eNodeB[m],model.weightB[wIndex]:transpose(1,2))

	 end
	 
	 --model.eNode[m-1]:addmm(model.eNodeB[m],model.weightB[wIndex]:transpose(1,2))
	 
	 m = m - 1
	 wIndex = wIndex - 1
	 
      end
   end
   
end

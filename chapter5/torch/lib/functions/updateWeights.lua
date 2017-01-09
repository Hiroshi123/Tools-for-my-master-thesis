

function updateWeights()
   
   --update weights
   
   local wIndex    = 1
   local nodeIndex = 1 - 1
   
--[[
   for k = 1, model.len-5 do
      print(k,"th node")
      print(model.node[k][1]:size())
   end

   for k = 1, model.len-5 do
      print(model.eNode[k][1]:size())
   end
--]]

   for k = 1,model.len do
      
      if (model.operations[k] == "convolution") then
	 
	 model.dWeight[wIndex]:fill(0)
	 
	 local inputF  = model.node[nodeIndex]:size()[2]
	 local outputF = model.eNode[nodeIndex]:size()[2]
	 local resW    = model.eNode[nodeIndex]:size()[3]
	 local resH    = model.eNode[nodeIndex]:size()[4]
	 
	 if(opt.GPU >= 0) then 
	    func1 = cudnn.SpatialConvolution(1,outputF,
				       resW,
				       resH,
				       model.kernel.width,
				       model.kernel.height,
				       model.kernel.width,
				       model.kernel.height,
				       1,1):cuda()
	 else
	    func1 = nn.SpatialConvolution(1,outputF,
				       resW,
				       resH,
				       model.kernel.width,
				       model.kernel.height,
				       model.kernel.width,
				       model.kernel.height,
				       1,1)
	    
	 end
	 
	 for i=1,inputF do
	    
	    --print(func1.weight)
	    --print(model.eNode[nodeIndex][1]:size())
	    
	    func1.weight = torch.reshape
	    (
	       model.eNode[nodeIndex][1],
	       outputF,
	       1,
	       resW,
	       resH
	    )
	    --print(func1.weight:size())
			   
	    model.dWeight[wIndex][i] = 
	       func1:forward
	       (torch.reshape(model.node[nodeIndex][1][i],1,
			   resW,
			   resH))
	       
	 end
	 
         
	 --model.dWeight[wIndex]:transpose(3,4)
	 
	 --print(model.dWeight[wIndex]:size())
	 
	 model.weight[wIndex]:add(opt.lr,model.dWeight[wIndex])
	 
	 --model.weight[wIndex]:csub(opt.lr,model.dWeight[wIndex])
	 
	 wIndex = wIndex + 1
	 
      end
      
      if (model.operations[k] == "linear") then
	 local inputF  = model.weight[wIndex]:size()[1]
	 local outputF = model.weight[wIndex]:size()[2]
--	 print(inputF,outputF)
--	 print(model.node[nodeIndex]:size())
--	 print(model.eNode[nodeIndex]:size())
	 model.dWeight[wIndex]:fill(0)
	 if (opt.GPU >= 0) then
	    local dot = nn.Linear(inputF,outputF):cuda()
	    --model.dWeight[wIndex] = nn.Linear(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex]):cuda()
	    model.dWeight[wIndex]:addmm(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex]):cuda()
	    --dot:forward(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex])
	    --model.dWeight[wIndex] = dot:forward(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex])
	    currentLr = currentLr - opt.lrDecay
	    local lr = torch.Tensor(model.dWeight[wIndex]:size()):fill(currentLr):cuda()
	    model.weight[wIndex]:add(lr,model.dWeight[wIndex]):cuda()
	    
	 else
	    local dot = nn.Linear(inputF,outputF)
	    
	    model.dWeight[wIndex]:addmm(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex])
	    --model.dWeight[wIndex] = dot:forward(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex]:transpose(1,2))
	    --print(model.weight[wIndex]:size())
	    currentLr = currentLr - opt.lrDecay
	    model.weight [wIndex]:add(currentLr,model.dWeight[wIndex])
	 end

	 --model.dWeight[wIndex]:addmm(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex])
	 --nodeIndex = nodeIndex + 1
	 wIndex = wIndex + 1
	 
      end
      
      if (model.operations[k] == "relu" or model.operations[k] == "sigmoid") then
      else
	 nodeIndex = nodeIndex + 1
      end
      
   end
end

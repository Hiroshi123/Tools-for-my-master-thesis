
--this function is for memory allocation in static memory
--network architecture which you have defined in your model file could be placed in your computer with this function


function initMemoryAllocation()
   
   local count = 0
   local wCount = 1
   
   poolingMaxIndex = 0
   
   for k,v in pairs(model.operations) do
      
      if(v == "setData") then
	 if(opt.GPU >= 0) then 	 
	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    --print(cudnn.typemap[torch.typename.model.node[1]])
	 else
	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 end
	 --Count = nodeCount + 1
      end
      
      if(v == "setData3D") then
	 if(opt.GPU >= 0) then
	    table.insert(model.node,torch.Tensor (opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	 else 
	    table.insert(model.node,torch.Tensor (opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 end

	 --Count = nodeCount + 1
      end
      
      if(v == "linear") then
	 if(opt.GPU >= 0) then 

	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.weight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	    table.insert(model.weightB,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0):cuda())
	    table.insert(model.dWeight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0):cuda())
	 
	 else 
	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.weight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	    table.insert(model.weightB,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	    table.insert(model.dWeight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	 
	 end 
	 model.weight[wCount] = torch.rand(model.layerN[k-1],model.layerN[k]) *  2 - 1
	 if(opt.GPU >= 0) then
	    model.weight[wCount] = model.weight[wCount]:cuda()
	 end
	 
	 wCount = wCount + 1
	 
      end
      
      if( v == "convolution") then
	 if(opt.GPU >= 0) then 
	    
	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.weight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	    table.insert(model.weightB,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0):cuda())
	    table.insert(model.dWeight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0):cuda())
	    
	 else
	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.weight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	    table.insert(model.weightB,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	    table.insert(model.dWeight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	 end
	 
	 model.weight[wCount] = torch.rand(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height) *  2 - 1
	 
	 if(opt.GPU >= 0) then 
	    model.weight[wCount] = model.weight[wCount]:cuda()
	 end
	 	 
	 wCount = wCount + 1
	 
      end
      
      if( v == "maxPooling") then

	 if(opt.GPU >= 0) then 
	    
	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0):cuda())

	 else

	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 
	 end
	 poolingMaxIndex = poolingMaxIndex + 1

	 --note here that no weights should be inserted
	 
      end

      if(v == "reshape") then
	 if(opt.GPU >= 0) then 

	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0):cuda())
	    
	 else
	    
	    table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	    table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))

	 end

      end
      
      count = k
      
   end

   if(opt.GPU >= 0) then 
      model.teacher = torch.Tensor(opt.bs,model.layerN[count]):fill(0):cuda()  
   else
      model.teacher = torch.Tensor(opt.bs,model.layerN[count]):fill(0)
   end

end

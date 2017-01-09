
require 'nn'
require 'mnist'
require 'paths'
require 'lib/util'
require 'optim'
require 'gnuplot'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.0002,'learning rate')
cmd:option('-bs',1,'batch size')
cmd:option('-ac',100,'accuracy measurement interval')
cmd:option('-ds','cifar10','data set')
cmd:option('-model','model4','model name')
cmd:option('-epoch',300,'epoch times')
cmd:option('-dir','results','a name of output directory')
cmd:option('-threads',1,'number of threads')
cmd:option('-plot',1,'plot')
cmd:option('-title','Accuracy per epoch','title for chart')
cmd:option('-usingSampleN',100,'how many data among dataset you are going to iterate')
cmd:text()

opt = cmd:parse(arg)


if(opt.ds == "mnist") then
   print("mnist is loaded")
   require './load/mnist'
end

if(opt.ds == "cifar10") then
   print("cifar10 is loaded")
   require './load/cifar10'
end

if(opt.ds == "cifar100") then
   print("cifar100 is loaded")
   require './load/cifar100'
end


os.execute('mkdir -p ' .. opt.dir)

opt.dir = paths.concat('./', opt.dir)

cmd:log(opt.dir .. '/' .. opt.ds .. '_' .. opt.model .. '.txt', opt)

-- threads
torch.setnumthreads(opt.threads)
--print('<torch> set nb of threads to ' .. torch.getnumthreads())

torch.manualSeed(123)

--torch.seed()
--package.path=package.path..';./?.lua'

local pathForModel = ("./" .. "model/" .. opt.model)
require (pathForModel)

torch.setdefaulttensortype('torch.FloatTensor')

--this function is for memory allocation in static memory
--network architecture which you have defined in your model file could be placed in your computer with this function

function initMemoryAllocation()
   
   local count = 0
   local wCount = 1

   poolingMaxIndex = 0
   
   for k,v in pairs(model.operations) do
      
      if(v == "setData") then
	 table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 --Count = nodeCount + 1
      end
      
      if(v == "setData3D") then
	 table.insert(model.node,torch.Tensor (opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 --Count = nodeCount + 1
      end
      
      if(v == "linear") then
	 
	 table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.weight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	 table.insert(model.weightB,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	 table.insert(model.dWeight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	 
	 model.weight[wCount] = torch.rand(model.layerN[k-1],model.layerN[k]) *  2 - 1
	 wCount = wCount + 1
	 
      end
      
      if( v == "convolution") then
	 
	 table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.weight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	 table.insert(model.weightB,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	 table.insert(model.dWeight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	 
	 model.weight[wCount] = torch.rand(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height) *  2 - 1
	 wCount = wCount + 1
	 
      end
      
      if( v == "maxPooling") then
	 
	 table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k][1],model.layerN[k][2],model.layerN[k][3]):fill(0))
	 
	 poolingMaxIndex = poolingMaxIndex + 1
	 
	 --table.insert(model.weight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	 --table.insert(model.weightB,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	 --table.insert(model.dWeight,torch.Tensor(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height):fill(0))
	 
	 
      end

      if(v == "reshape") then
	 
	 table.insert(model.node,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.nodeT,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.eNode,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 table.insert(model.eNodeB,torch.Tensor(opt.bs,model.layerN[k]):fill(0))
	 
      end
      
      count = k
      
   end
   
   model.teacher = torch.Tensor(opt.bs,model.layerN[count]):fill(0)
   
end

function feedForward(i,train_or_test)
   
   local nodeIndex   = 1
   local weightIndex = 1
   
   poolingFuncTable = {}
   
   
   for k = 1,model.len do
      if ( model.operations[k] == "setData" or model.operations[k] == "setData3D" ) then
	 for b = 1,opt.bs do
	    model.node[1][b]:fill(0)
	    if(train_or_test) then
	       --print(model.node[1][b]:size())
	       --print(dataset.trainInput[i]:size())
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
	 
	 
	 local conv = nn.SpatialConvolution(model.layerN[k-1][1],model.layerN[k][1],model.kernel.width,model.kernel.height,1,1,1,1)
	 --print(model.weight[weightIndex]:size())
	 --conv.weight = model.weight[weightIndex]:transpose(1,2)
	 conv.weight = model.weightB[weightIndex]:transpose(1,2)
	 --model.node[nodeIndex+1] = conv:forward(model.node[nodeIndex])
	 model.node[nodeIndex+1] = conv:forward(model.nodeB[nodeIndex])
	 nodeIndex   = nodeIndex + 1
	 weightIndex = weightIndex + 1
	 
      end
      
      if (model.operations[k] == "maxPooling") then
	 
	 pooling1 = nn.SpatialMaxPooling(2,2)
	 table.insert(poolingFuncTable,pooling1)
	 model.node[nodeIndex+1] = pooling1:forward(model.node[nodeIndex])
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
	 model.node[nodeIndex+1]:addmm(model.nodeB[nodeIndex],model.weightB[weightIndex])
	 
	 nodeIndex   = nodeIndex + 1
	 weightIndex = weightIndex + 1
	 
      end
      
      if (model.operations[k] == "relu") then
	 
	 model.node [nodeIndex]   = nn.ReLU():forward(model.node[nodeIndex])
	 model.nodeT[nodeIndex-1] = model.node[nodeIndex]:clone()
	 
	 --model.nodeT[nodeIndex-1]:apply(deSigmoid)
	 
      end
      
      if (model.operations[k] == "sigmoid") then
	 model.node[nodeIndex] = nn.Sigmoid():forward(model.node[nodeIndex])
	 model.nodeT[nodeIndex-1] = model.node[nodeIndex]:clone()
	 model.nodeT[nodeIndex-1]:apply(deSigmoid)
      end
      
   end
   
end

function calculateError(i,train_or_test)
   
   --output data setting
   for b = 1,opt.bs do
      model.teacher[b]:fill(0)
      if(train_or_test) then
	 model.teacher[b][dataset.trainTeacher[i]+1] = 1
      else
	 model.teacher[b][dataset.testTeacher[i]+1] = 1
      end
   end
   
   model.eNode[model.nodeLen-1] = model.teacher:csub(model.node[model.nodeLen])
   --model.teacher:csub(model.node[model.nodeLen])
   
   local errors = 0
   errors = torch.sum(torch.abs(model.eNode[model.nodeLen-1]))
   --print(errors)
   
end

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
	 
	 local conv2  = nn.SpatialConvolution(model.layerN[k][1],model.layerN[k-1][1],model.kernel.width,model.kernel.height,1,1,1,1)
	 
	 
	 --conv2.weight = model.weight[wIndex]--:transpose(1,2)
	 --model.eNode[m-1] = conv2:forward (model.eNode[m])
	 conv2.weight = model.weightB[wIndex]--:transpose(1,2)
	 model.eNode[m-1] = conv2:forward (model.eNodeB[m])
	 m = m - 1
	 wIndex = wIndex - 1
	 
      end
      
      if (model.operations[k] == "maxPooling") then
	 model.eNode[m-1]:fill(0)
	 unPooling1 = nn.SpatialMaxUnpooling(poolingFuncTable[poolingIndex]) --(pooling1)
	 model.eNode[m-1] = unPooling1:updateOutput(model.eNode[m])
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
	 
	 
	 model.eNode[m-1]:addmm(model.eNodeB[m],model.weightB[wIndex]:transpose(1,2))
	 
	 m = m - 1
	 wIndex = wIndex - 1
	 
      end
   end
   
end



function updateWeights()
   
   --update weights
   
   local wIndex    = 1
   local nodeIndex = 1 - 1
   
   
   for k = 1,model.len do
      
      if (model.operations[k] == "convolution") then

	 
	 model.dWeight[wIndex]:fill(0)

	 --very diirty alternative method for updating weights which is not used for convolution,
	 --but element-wise multiplication and sliding operation
	 
	 --[[
	 for w = 1,model.kernel.width do
	    for h = 1,model.kernel.height do
	       for i=1,model.node[nodeIndex]:size()[2] do
		  for o=1,model.eNode[nodeIndex]:size()[2] do
		     
		     tt = 0
		     
		     --if(o == 8) then
		     --print(model.node[nodeIndex][1][i])
		     --print(model.eNode[nodeIndex][1][o])
		     --end
		     
		     if (w == 1 and h == 1) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{1,-2},{1,-2}}],
						 model.eNode[nodeIndex][1][o][{{2,-1},{2,-1}}]))
		     end
		     if ( w == 2 and h == 1 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{1,-1},{1,-2}}],
						 model.eNode[nodeIndex][1][o][{{1,-1},{2,-1}}]))
		     end
		     if ( w == 3 and h == 1 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{2,-1},{1,-2}}],
						 model.eNode[nodeIndex][1][o][{{1,-2},{2,-1}}]))
		     end	
		     if ( w == 1 and h == 2 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{1,-2},{1,-1}}],
						 model.eNode[nodeIndex][1][o][{{2,-1},{1,-1}}]))
		     end
		     
		     if ( w == 2 and h == 2 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{1,-1},{1,-1}}],
						 model.eNode[nodeIndex][1][o][{{1,-1},{1,-1}}]))
			--print("come")
		     end
		     
		     if ( w == 3 and h == 2 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{2,-1},{1,-1}}],
						 model.eNode[nodeIndex][1][o][{{1,-2},{1,-1}}]))
		     end
		     if ( w == 1 and h == 3 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{1,-2},{2,-1}}],
						 model.eNode[nodeIndex][1][o][{{2,-1},{1,-2}}]))
		     end
		     
		     if ( w == 2 and h == 3 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{1,-1},{2,-1}}],

						  model.eNode[nodeIndex][1][o][{{1,-1},{1,-2}}]))
		     end
		     
		     if ( w == 3 and h == 3 ) then
			tt = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{2,-1},{2,-1}}],
						 model.eNode[nodeIndex][1][o][{{1,-2},{1,-2}}]))
			
		     end
		     
		     --local t = torch.sum(torch.cmul(model.node[nodeIndex][1][i][{{1+(w-1),-1+(w-1)}},{{1+(w-1),-1}}],
		     --			    model.eNode[nodeIndex][1][o][{{1,-1}},{{1,-1}}]))
		     --print(tt)
		     
		     
		     if (tt > 100 or tt < -100) then
			print("AAAA",tt)
			print(model.node[nodeIndex][1][i])
			print(model.eNode[nodeIndex][1][o])
			--print(w,h,i,o)
			break
		     end
		     --model.dWeight[wIndex][i][o][w][h] = 0
		     model.dWeight[wIndex][i][o][w][h] = tt
		  end
	       end
	    end
	 end
	 --]]
	 --model.node [nodeIndex]:size(2) == inputNumber
	 --model.eNode[nodeIndex]:size(2) == outputNumber
	 
	 
	 model.dWeight[wIndex]:fill(0)
	 
	 local inputF  = model.node[nodeIndex]:size()[2]
	 local outputF = model.eNode[nodeIndex]:size()[2]
	 local resW    = model.eNode[nodeIndex]:size()[3]
	 local resH    = model.eNode[nodeIndex]:size()[4]
	 
	 func1 = nn.SpatialConvolution(1,outputF,
				       resW,
				       resH,
				       model.kernel.width,
				       model.kernel.height,
				       model.kernel.width,
				       model.kernel.height,
				       1,1)
	 
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

	 model.dWeight[wIndex]:fill(0)
	 
	 model.dWeight[wIndex]:addmm(model.node[nodeIndex]:transpose(1,2),model.eNode[nodeIndex])
	 model.weight [wIndex]:add(opt.lr,model.dWeight[wIndex])
	 
	 --nodeIndex = nodeIndex + 1
	 wIndex = wIndex + 1
	 
      end
      
      if (model.operations[k] == "relu" or model.operations[k] == "sigmoid") then
      else
	 nodeIndex = nodeIndex + 1
      end
      
   end
end

nexts = false

sampleN = opt.usingSampleN

function train(epochN)
   
   local traiN = true

   --if(nexts) then
   --   randomData = torch.randperm(200)
   --else

   randomData = torch.randperm(sampleN)
   
   --end
   
   --local randomData = torch.randperm(dataset.trainTeacher:size()[1])
   
   for e = 1,epochN do
      local indexOrder = randomData
      --local indexOrder = randomData[{{1,opt.ac}}]
      --local s = indexOrder:storage()
      local correct = 0
      for i=1,opt.ac,opt.bs do
	 --if (sampleN < opt.ac) then
	 --   sample = indexOrder[math.fmod(i,sampleN)+1]
	 --else
	 sample = indexOrder[i]
	 --end
	 --sample = i
	 --print("sample",sample)
	 feedForward(sample,traiN)
	 calculateError(sample,traiN)
	 local temp,maxIndex = torch.max(model.node[model.nodeLen],2)
	 --print(maxIndex[1][1],dataset.trainTeacher[i]+1)
	 for b = 1,opt.bs do
	    if (maxIndex[1][b] == dataset.trainTeacher[sample]+1) then
	       correct = correct + 1
	    end
	 end
	 
	 feedBack(sample)
	 updateWeights(sample)
	 
      end
      print("train accuracy = " , correct / opt.ac)
      table.insert(trainAccuracy,correct / opt.ac)
      
      --print(model.eNode[2][1])
      if(correct > 95) then
	 sampleN = sampleN + 10
	 print("sample",sampleN)
      end
      
	 
   end
   
end

function test()
   
   local traiN = false
   local correct = 0
   randomData = torch.randperm(dataset.testTeacher:size()[1])
   for i=1,opt.ac,opt.bs do
      feedForward(randomData[i],trainN)
      local temp,maxIndex = torch.max(model.node[model.nodeLen],2)
      for b = 1,opt.bs do
	 --print(maxIndex[1][b],dataset.testTeacher[i]+1)
	 if (maxIndex[1][b] == dataset.testTeacher[randomData[i]]+1) then
	    correct = correct + 1
	 end
      end
   end
   
   print("test accuracy = " , correct / opt.ac)
   
   table.insert(testAccuracy,correct / opt.ac)
   
end

print("---main---")

setData()
setModel()
initMemoryAllocation()

trainAccuracy = {}
testAccuracy  = {}


for e = 1,opt.epoch do
   
   train(1)
   test()
   
end

if opt.plot == 1 then
   
   gnuplot.figure(1)
   gnuplot.title(opt.title)
   
   local trainA = torch.Tensor(opt.epoch)
   local testA  = torch.Tensor(opt.epoch)
   
   for e = 1,opt.epoch do
      trainA[e] = trainAccuracy[e]
      testA[e]  = testAccuracy[e]
   end
   
   gnuplot.plot({'train',trainA,'-'},{'test',testA,'-'})
   gnuplot.xlabel('Epoch (times)')
   gnuplot.ylabel('Accuracy')
   gnuplot.plotflush() 
   
end


print("end")

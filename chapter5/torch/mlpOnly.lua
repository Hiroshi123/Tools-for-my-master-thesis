
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
cmd:option('-lr',0.1,'learning rate')
cmd:option('-bs',1,'batch size')
cmd:option('-ac',100,'accuracy measurement interval')
cmd:option('-ds','mnist','data set')
cmd:option('-model','model1','model name')
cmd:option('-epoch',10,'epoch times')
cmd:option('-dir','results','a name of output directory')
cmd:option('-threads',1,'number of threads')
cmd:option('-plot',1,'plot')
cmd:option('-title','Accuracy per epoch','title for chart')
cmd:option('-fname','chart01','title for chart')
cmd:text()

opt = cmd:parse(arg)

os.execute('mkdir -p ' .. opt.dir)

opt.dir = paths.concat('./', opt.dir)

cmd:log(opt.dir .. '/' .. opt.ds .. '_' .. opt.model .. '.txt', opt)

-- threads
torch.setnumthreads(opt.threads)
--print('<torch> set nb of threads to ' .. torch.getnumthreads())

torch.manualSeed(123)

--torch.seed()
--package.path=package.path..';./?.lua'

if(opt.ds == "mnist") then
   require './load/mnist'
end

if(opt.ds == "cifar10") then
   require './load/cifar10'
end

local pathForModel = ("./" .. "model/" .. opt.model)
require (pathForModel)

function initMemoryAllocation()
   
   local count = 0
   local wCount = 1
   
   for k,v in pairs(model.operations) do
      print(k,v,model.layerN[k])
      if(v == "setData") then
	 table.insert(model.node,torch.Tensor(1,model.layerN[k]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(1,model.layerN[k]):fill(0))
	 --Count = nodeCount + 1
      end
      
      if(v == "linear")  then
	 
	 table.insert(model.node,torch.Tensor(1,model.layerN[k]):fill(0))
	 table.insert(model.nodeB,torch.Tensor(1,model.layerN[k]):fill(0))
	 table.insert(model.nodeT,torch.Tensor(1,model.layerN[k]):fill(0))
	 table.insert(model.eNode,torch.Tensor(1,model.layerN[k]):fill(0))
	 table.insert(model.eNodeB,torch.Tensor(1,model.layerN[k]):fill(0))
	 table.insert(model.weight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	 table.insert(model.weightB,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	 table.insert(model.dWeight,torch.Tensor(model.layerN[k-1],model.layerN[k]):fill(0))
	 
	 model.weight[wCount] = torch.rand(model.layerN[k-1],model.layerN[k]) *  2 - 1
	 wCount = wCount + 1
	 
      end
      
      count = k
      
   end
   
   model.teacher = torch.Tensor(1,model.layerN[count]):fill(0)   
   
end

function feedForward(i,train_or_test)
   
   --   for i=1,s:size() do
   
   local nodeIndex = 1
   for k = 1,model.len do
      if(model.operations[k] == "setData") then
	 for b = 1,opt.bs do
	    model.node[1][b]:fill(0)
	    if(train_or_test) then
	       model.node[1][b] = dataset.trainInput[i]
	       
	    else
	       model.node[1][b] = dataset.testInput[i]
	    end
	 end
	 --nodeIndex = nodeIndex + 1
      end
      
      if(model.operations[k] == "linear") then
	 model.node[nodeIndex+1]:fill(0)
	 --binarize
	 --
	 --model.weightB[nodeIndex] = model.weight[nodeIndex]:clone()
	 
	 if (model.bitForward[nodeIndex] == 32) then
	    model.nodeB[nodeIndex] = model.node[nodeIndex]:clone()
	 else
	    local t1 = torch.pow(2,model.bitForward[nodeIndex])
	    model.nodeB[nodeIndex] = torch.floor(model.node[nodeIndex] * t1) / t1
	 end
	 
	 if (model.bitWeight[nodeIndex] == 32) then
	    model.weightB[nodeIndex] = model.weight[nodeIndex]:clone()
	 else
	    local t2 = torch.pow(2,model.bitWeight[nodeIndex])
	    model.weightB[nodeIndex] = torch.floor((model.weight[nodeIndex] + 1) * ( t2 / 2 ) ) * (4 / t2) - 1
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
	 
	 model.node[nodeIndex+1]:addmm(model.nodeB[nodeIndex],model.weightB[nodeIndex])
	 nodeIndex = nodeIndex + 1
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

   local m = model.nodeLen-1
   
   for k = model.len-1,3,-1 do
      if(model.operations[k] == "sigmoid") then
	 model.eNode[m]:cmul(model.nodeT[m])
      end
      
      if(model.operations[k] == "linear") then
	 -- model.weightB[m] --= model.weight[m]:apply(binarize)
	 -- model.eNodeB[m] = model.eNode[m]:clone()
	 if (model.bitBackward[m] == 32) then
	    model.eNodeB[m] = model.eNode[m]:clone()
	 else
	    local t3 = torch.pow(2,model.bitBackward[m])
	    model.eNodeB[m] = torch.floor((model.eNode[m] + 1) * (t3 / 2 )) * ( 4 / t3 ) - 1
	 end
	 
	 --if(model.bitBackward[m] == 1) then
	 --   model.eNodeB[m]:apply(binarize)
	 --end
	 --if (model.bitBackward[m] == 2) then
	 --   model.eNodeB[m]:apply(binarize2)
	 --end
	 
	 model.eNode[m-1]:addmm(model.eNodeB[m],model.weightB[m]:transpose(1,2))
	 m = m - 1
	 
      end      
   end
   
end

function updateWeights()
   
   --update weights
   
   -- for i = 1,model.weightLen do
   --    local t1 = torch.pow(2,3)
   --    model.node[i] = torch.floor(model.node[i] * t1) / t1
   --    local t3 = torch.pow(2,3)
   --    model.eNode[i] = torch.floor((model.eNode[i] + 1) * (t3 / 2 )) * ( 4 / t3 ) - 1
   --    --(model.node[i]):apply(ex2)
   -- end
   
   -- for i = 1,model.weightLen do
   --    (model.eNode[i]):apply(ex1)
   -- end
   
   
   for i = 1,model.weightLen do
         
      model.dWeight[i]:fill(0)
      --model.dWeight[i]:addmm(model.node[i]:transpose(1,2),model.eNode[i])
      model.dWeight[i]:addmm(model.nodeB[i]:transpose(1,2),model.eNodeB[i])
      model.weight[i]:add(opt.lr,model.dWeight[i])
      
   end
   
end

function train(epochN)
   local traiN = true
   local randomData = torch.randperm(dataset.trainTeacher:size()[1])
   for e = 1,epochN do
      local indexOrder = randomData[{{1,opt.ac}}]
      --local s = indexOrder:storage()
      local correct = 0
      for i=1,opt.ac,opt.bs do
	 local sample = indexOrder[i]
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
      
   end
end

function test()
   local traiN = false
   local correct = 0
   local randomData = torch.randperm(dataset.testTeacher:size()[1])
   for i=1,opt.ac,opt.bs do
      feedForward(i,trainN)
      local temp,maxIndex = torch.max(model.node[model.nodeLen],2)
      for b = 1,opt.bs do
	 --print(maxIndex[1][b],dataset.testTeacher[i]+1)
	 if (maxIndex[1][b] == dataset.testTeacher[i]+1) then
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
   
   gnuplot.pngfigure(opt.fname .. '.png')
   gnuplot.plot({'train',trainA,'-'},{'test',testA,'-'})
   gnuplot.xlabel('Epoch (times)')
   gnuplot.ylabel('Accuracy')
   gnuplot.plotflush()
   
end


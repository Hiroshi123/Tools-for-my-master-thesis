
slideW = 0

function train(epochN)
   
   local traiN = true

   --if(nexts) then
   --   randomData = torch.randperm(200)
   --else

   randomData = torch.randperm(sampleN) + slideW
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
      --[[
      if(correct > 95) then
	 sampleN = sampleN + 10
	 print("sample",sampleN)
      end
      --]]
      
   end

   slideW = slideW + opt.slideW
   
end

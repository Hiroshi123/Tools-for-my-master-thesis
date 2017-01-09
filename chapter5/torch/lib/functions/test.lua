
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

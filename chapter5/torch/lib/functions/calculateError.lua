

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
   --print(model.node[model.nodeLen][1])
   --print(errors)
   
end

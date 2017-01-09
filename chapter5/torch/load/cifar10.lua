
dataset = {}

function setData()
      
   local trainset = torch.load('./load/data/cifar10-train.t7')
   local testset  = torch.load('./load/data/cifar10-test.t7')

   local train_input = trainset.data[{{1,50000}}]:float()
   local test_input  = testset.data[{{1,10000}}]:float()
   
   train_input = train_input / 255.0
   test_input  = test_input / 255.0
   
   dataset = {
      trainInput   = train_input,
      trainTeacher = trainset.label[{{1,50000}}]:float(),
      testInput    = test_input,
      testTeacher  = testset.label[{{1,10000}}]:float()
      
   }
end


		    

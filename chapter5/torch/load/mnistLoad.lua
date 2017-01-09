
dataSet = {}

function mnistSet()
   
   local mnist = require 'mnist'
   local trainset = mnist.traindataset()
   local testset = mnist.testdataset()
   local train_input = trainset.data[{{1,60000}}]:float()
   train_input = train_input / 255.0
   
   local test_input = testset.data[{{1,10000}}]:float()
   test_input = test_input / 255.0
   
   dataset = {
      trainInput   = train_input,
      trainTeacher = trainset.label[{{1,60000}}]:float(),
      testInput    = test_input,
      testTeacher  = testset.data[{{1,10000}}]:float()     
   }
   
end



require 'nn'
require 'mnist'
require 'paths'

require 'optim'
require 'gnuplot'

--main logic of neural network
require './lib/functions/memoryAllocate'
require './lib/functions/feedForward'
require './lib/functions/calculateError'
require './lib/functions/feedBack'
require './lib/functions/updateWeights'
require './lib/functions/train'
require './lib/functions/test'

require 'lib/util'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.01,'learning rate')
cmd:option('-lrDecay',0.0000001,'learning rate')
cmd:option('-bs',1,'batch size')
cmd:option('-ac',100,'accuracy measurement interval')
cmd:option('-ds','cifar10','data set')
cmd:option('-model','model4','model name')
cmd:option('-epoch',1000,'epoch times')
cmd:option('-dir','results','a name of output directory')
cmd:option('-threads',8,'number of threads')
cmd:option('-plot',1,'plot')
cmd:option('-title','Accuracy per epoch','title for chart')
cmd:option('-usingSampleN',100,'how many data among dataset you are going to iterate')
cmd:option('-imageFileExtension','svg','image file extension, among png,svg,eps,pdf')
cmd:option('-chartFileName','chart1','when you save a plot, what do you call it?')
cmd:option('-GPU',0,'if you make use of gpu or not')
cmd:option('-slideW',100,'slide to the next dataset')
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

--torch.setdefaulttensortype('torch.FloatTensor')

sampleN = opt.usingSampleN

trainAccuracy = {}
testAccuracy  = {}

currentLr = opt.lr

print("---main---")

function main()
   
   setData()
   setModel()
   initMemoryAllocation()
   
   --train(1)
   
   for e = 1,opt.epoch do
   
      train(1)
      test()
   
   end
   
   if opt.plot == 1 then
      require './lib/plot'
      plotChart()
   end
   
   
end

if(opt.GPU == 0) then
   require 'cutorch'
   require 'cunn'
   require 'cudnn'
end

main()

print("end")

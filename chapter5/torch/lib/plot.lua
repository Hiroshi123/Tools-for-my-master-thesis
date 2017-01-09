
require 'gnuplot'


function plotChart()    

   gnuplot.figure(1)
   gnuplot.title(opt.title)
   
   local trainA = torch.Tensor(opt.epoch)
   local testA  = torch.Tensor(opt.epoch)
   
   for e = 1,opt.epoch do
      trainA[e] = trainAccuracy[e]
      testA[e]  = testAccuracy[e]
   end
   
   if(opt.imageFileExtension == "png") then      
      gnuplot.pngfigure("./chart/" .. opt.chartFileName .. ".png")
   end   
   
   if(opt.imageFileExtension == "eps") then      
      gnuplot.pngfigure("./chart/" .. opt.chartFileName .. ".eps")
   end

   if(opt.imageFileExtension == "pdf") then
      gnuplot.epsfigure("./chart/" .. opt.chartFileName .. ".pdf")
   end

   if(opt.imageFileExtension == "svg") then
      gnuplot.svgfigure("./chart/" .. opt.chartFileName .. ".svg")
   end
   
   gnuplot.plot({'train',trainA,'-'},{'test',testA,'-'})
   gnuplot.xlabel('Epoch (times)')
   gnuplot.ylabel('Accuracy')
   gnuplot.plotflush()

end

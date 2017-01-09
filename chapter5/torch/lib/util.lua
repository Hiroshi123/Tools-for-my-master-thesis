

function myRelu(arg)
   if arg > 0 then return arg
   else return 0
   end
end

function deSigmoid(arg)
   return arg * ( 1 - arg )
end

function binarizeP(arg)
   if arg > 0.5 then return 1
   else return 0
   end
end

function binarizeP2(arg)
   if arg > 0.75 then return 1 end
   if arg > 0.5  then return 0.67 end
   if arg > 0.25 then return 0.33 end
   return 0
end

function binarizeP3(arg)
   if arg > 0.75 then return 1 end
   if arg > 0.5  then return 0.67 end
   if arg > 0.25 then return 0.33 end
   return 0
end

function binarize(arg)
   if arg > 0 then return 1 end
   return -1
end


function binarize2(arg)
   if arg > 0.5  then return 1 end
   if arg > 0    then return 1  - 0.66 end
   if arg > -0.5 then return -1 + 0.66 end
   return -1
end

function binarizeG(arg)
   return torch.floor(arg * 8) / 8
end

function floor_ceil(arg)
   if arg > 0 then return torch.ceil(arg) end
   return torch.floor(arg)
end


function ex1(arg)
   if arg > 0.5 then return 0.1
   elseif arg < -0.5 then return -0.1
   else
      return 0
   end
end

function ex2(arg)
   if arg > 0.5 then return 1
   else
      return 0
   end
end

   

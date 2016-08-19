{-# LANGUAGE MultiParamTypeClasses #-}


import MyLib.General
import MyLib.Matrix
import MyLib.MyMath
import MyLib.MyComplex
import Convolve

type KernelSize = Int

type Node1D = [[Double]]

type Error1D = [[Double]]

type Weight1D = [[[Double]]]

type Node2D = [[[Double]]]

type Error2D = [[[Double]]]

type Weight2D = [[[[Double]]]]

type Teacher = [Double]

type LearningRate = Double


class CNN where
  
  --argument
  --1st : activation function
  --2nd : previous node
  --3rd : weights in between
  --return : Node on next layers
  feedForward1D :: (Double->Double) -> Node1D -> Weight1D -> Node1D
  
  feedForward1D' :: (Double->Double) -> Node1D -> Weight1D -> [Node1D]
  
  --dimension is one higher than 1D
  feedForward2D :: (Double->Double) -> Node2D -> Weight2D -> Node2D
  
  --feedForward2D' :: (Double->Double) -> Node2D -> Weight2D -> Node2D
  
  
  --rectified linear unit
  relu :: Double -> Double
  
  --derivative of rectified linear unit
  deRelu :: Double -> Double
  
  --argument
  --1st : node
  --return : node
  maxPooling1D  :: Node1D -> Node1D
  
  --argument
  --1st : pooling size (default 2)
  --2nd : node
  --return : node
  maxPooling1D' :: Int -> Node1D -> Node1D

  --same with 1D
  maxPooling2D  :: Node2D -> Node2D

  --same with 1D
  maxPooling2D' :: Int -> Node2D -> Node2D

  --argument
  --1st : teacher
  --2nd : nodes on last layer
  -- return : difference between teacher and node
  calculateFinalErrors1D :: Teacher -> Node1D -> Error1D

  --same with 1D
  calculateFinalErrors2D :: Teacher -> Node2D -> Error2D

  --argument
  --1st : derivative of activation function
  --2nd : Node on previous layer
  --3rd : Error on subsequent layer
  --4th : weights in between
  backPropagateError1D :: (Double->Double) -> Node1D -> Error1D -> Weight1D -> Error1D

  --same with 1D
  backPropagateError2D :: (Double->Double) -> Node2D -> Error2D -> Weight2D -> Error2D
  
  --argument
  --1st : learning rate
  --2nd : derivative of weight
  --3rd : weight before update
  --4th : updated weight
  updateWeight1D :: LearningRate -> Weight1D -> Weight1D -> Weight1D
  
  --same with 1D
  updateWeight2D :: LearningRate -> Weight2D -> Weight2D -> Weight2D
  
  --argument
  --1st : Node on previous layer
  --2nd : Error on subsequent layer
  --3rd : Kernel size (this is necessary for sliding method which I adopt)
  --return : derivate of weight
  weightDerivative1D  :: Node1D -> Error1D -> KernelSize -> Weight1D
  
  --sub function
  --weightDerivative1D' :: Node1D -> Error1D -> Error1D
  
  --same with 1D
  weightDerivative2D  :: Node2D -> Error2D -> KernelSize -> Weight2D
  --sub function
  --weightDerivative2D' :: Node2D -> Error2D -> Error1D
  
  slideNode1D :: KernelSize -> Node1D -> [Node1D]
  
  slideNode2D  :: KernelSize -> Node2D -> [[Node2D]]
  slideNode2D' :: KernelSize -> Node2D ->  [Node2D]
  
  pickUp  :: [a] -> Int -> Int -> [a]
  
    
instance CNN where
  
  --second map from left will iterate different features on same nodes,
  --then, rightest map2 puts pairs of input and weights on convolution calculation,
  --then, the matrix is transposed so it can be added by element-wise.
  feedForward1D f n = map $ map g . t' . map2 convolve1D n
     where g = f . foldl (+) 0
           
               
  feedForward1D' f n2 w4 = map (\w3 -> (map.map) (foldl (+) 0) $ map (t') $ map2 (\w2 n1 -> map (\w1 -> map (*w1) n1) w2) w3 n2) w4
  
  f2 n = slideNode1D 3 n
  
     --where g = f . foldl (+) 0
         
                         
                         
  --the way not to use signature "$" is following
  --feedForward1D f n = map (map (f . foldl (+) 0)) . t' . map2 (convolve2D) n
  
  --only difference between 1D and 2D is two.
  --One is convolving by 2D and another is covering over it with one more "map"
  feedForward2D f n = map $ map g . t' . map2 convolve2D n
     where g = map (f . foldl (+) 0)
           
  --the way not to use signature "$" is following
  --feedForward2D f n = map (map (map (f . foldl (+) 0)) . t' . map2 (convolve2D) n)
  
  --relu
  relu a
    | a > 0     = a
    | otherwise = 0

  --derivative of Relu
  deRelu a
    | a > 0     = a
    | otherwise = 0
    
  --assume pooling size is 2
  maxPooling1D = maxPooling1D' 2

  --pooling is done by following.
  --first group list with every N elements such as (N==2) , [2,3,4,5]-> [[2,3],[4,5]]
  --then, calculating max value among them
  maxPooling1D' s = map (map (foldr max 0) . group s)

  --assume pooling size is 2
  maxPooling2D = maxPooling2D' 2
  
  --only difference between 1D and 2D pooling is again two.
  --One is grouping by 2D and another is covering over it with one more "map"
  maxPooling2D' s = map (map (map (foldr max 0)) . group2D s)
  
  --this just takes differnt between final node and teacher signal
  calculateFinalErrors1D t = map (map2 (-) t)

  --same with 1D
  calculateFinalErrors2D t = map (map (map2 (-) t))

  --back propagation step of CNN is very similar to forward step.
  --howebver, there is a difference in terms of multiplying activation function.
  --In backward process, input for activation function is previous values of nodes not the propagated error.
  --that is why (*df xx)) is used.
  --Then, propagated error is multiplied by it
  backPropagateError1D df n = map2 (\x -> map2 (\xx -> (*(df xx)) . foldl (+) 0) x . t' . map2 (convolve1D) n)

  --same with 1D except one more map2 and convolution by 2D
  backPropagateError2D df n = map2 (\x -> map2 (\xx -> map2 (\xxx -> (*(df xxx)) . foldl (+) 0) xx ) x . t' . map2 (convolve2D) n)

  --update weight can be done 
  updateWeight1D lr = (map2.map2.map2) (\x y-> x*y*lr)
  updateWeight2D lr = (map2.map2.map2.map2) (\x y-> x*y*lr)
  
  -- process to calculate derivatives of weights in each layer
  -- weights are updated by values of previous node and accumulated error in the next layer
  -- since forward process is convolution, weights needs to be calculated by iterating all of
  -- pairs of node and errors. This operation can be done by dot product.
  weightDerivative1D n e fs = map (\x -> dot' x (t' e) ) m 
      where s  = fs `div` 2
            m  = slideNode1D s n
            
            
  --weightDerivative1D' n = map (\x -> (map (\xx -> foldl (+) 0 $ map2 (*) x xx)) n)
  --weightDerivative1D' n e = dot' n (t' e)
  
  --basic idea is same with corresponding 1D version,
  --note m is going to be five dimension, and most inner word dimension should be concatenated
  --to be fed into the dot product.
  weightDerivative2D n e fs = (map.map) (\x -> dot' (map (concat) x) (map (concat) e) ) m
      where s  = fs `div` 2
            m  = slideNode2D s n
           
            
  --weightDerivative2D' n = map (\x -> (map (\xx -> foldl (+) 0 $ map2 (*) (concat x) (concat xx))) n)

  
  --first let size of list longer so as to reflect kernel size of the convolution.
  --then, pick up all of possible combinations of filters 
  slideNode1D k = map (\x-> map2 (pickUp x) [0..2*k] (reverse [0..2*k])) . map (boundry1D k)
  
                  
  --this function pick up part of list dropping first I element, and last j element.          
  pickUp a i j = (reverse . drop j . reverse . drop i) a

  
  --size of twoD node is going to be larger in any directions, reflecting convolution kernel,
  --then, cut out all of combinations of the candidate list,
  --for instance, if the kernel size is three,
  --nice rectangular nodes should be cut.
  --this step can be done by applying pickUp function both vertically and horizontically.
  slideNode2D  k = map (\x -> slideNode2D' k (t' x)) . slideNode2D' k
  
  --2D version of slideNode1D
  slideNode2D' k = map (\x-> map2 (pickUp x) [0..2*k] (reverse [0..2*k])) . map (boundry2D k)
  
    
main = do
  
  print $ feedForward1D relu [[1..10],[1..10]] [[[1..3],[1..3]],[[1..3],[1..3]]]
  
  
  --print $ maxPooling2D [[[1..10],[2..11],[1..10],[1..10]],[[1..10],[1..10],[4..13],[1..10]]]
  
  --print $ group2D 2 [[1,2],[2,3],[1,2],[2,3]]
  let k = 2
  
  --print $ weightDerivative1D [[1,2,3],[2,3,4]] [[1,2,3],[2,3,4]] 3
  
  
  --weightDerivative1D [[1,2,3],[2,3,4]] [[1,2,3],[2,3,4]] 3 --[[[1,2,3]]]
  --print $ updateWeight1D 
  
  print "hei"
  


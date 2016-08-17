{-# LANGUAGE MultiParamTypeClasses #-}


import MyLib.General
import MyLib.Matrix
import MyLib.MyMath
import MyLib.MyComplex
import Convolve


type Node1D = [[Double]]

type Error1D = [[Double]]

type Weight1D = [[[Double]]]


type Node2D = [[[Double]]]

type Error2D = [[[Double]]]

type Weight2D = [[[[Double]]]]

type Teacher = [Double]

class CNN where
  
  feedForward1D :: (Double->Double) -> Node1D -> Weight1D -> Node1D
  
  feedForward2D :: (Double->Double) -> Node2D -> Weight2D -> Node2D
  
  relu :: Double -> Double
  
  maxPooling1D  :: Node1D -> Node1D
  maxPooling1D' :: Int -> Node1D -> Node1D
  
  maxPooling2D  :: Node2D -> Node2D
  maxPooling2D' :: Int -> Node2D -> Node2D
  
  calculateFinalErrors1D :: Teacher -> Node1D -> Error1D
  
  calculateFinalErrors2D :: Teacher -> Node2D -> Error2D
  
  
instance CNN where
  
  feedForward1D f n = map ( map (f . foldl (+) 0) . t' . map2 (convolve1D) n)
  
  feedForward2D f n = map (map (map (f . foldl (+) 0)) . t' . map2 (convolve2D) n)
  
  relu a
    | a > 0     = a
    | otherwise = 0  
    
  maxPooling1D = maxPooling1D' 2
  
  maxPooling1D' s = map (map (foldr max 0) . group s)
    
  maxPooling2D = maxPooling2D' 2
  
  maxPooling2D' s = map (map (map (foldr max 0)) . group2D s)
  
  calculateFinalErrors1D t n = map (map2 (-) t) n

  calculateFinalErrors2D t n = map (map (map2 (-) t)) n
  
main = do
  
  print $ feedForward1D relu [[1..10],[1..10]] [[[1..3],[1..3]],[[1..3],[1..3]]]
  print $ maxPooling2D [[[1..10],[2..11],[1..10],[1..10]],[[1..10],[1..10],[4..13],[1..10]]]
  
  --print $ group2D 2 [[1,2],[2,3],[1,2],[2,3]]
  
  print "hei"
  


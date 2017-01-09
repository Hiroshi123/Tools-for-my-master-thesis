{-# LANGUAGE MultiParamTypeClasses #-}


import MyLib.General
import MyLib.Matrix
import MyLib.MyMath
import MyLib.MyComplex

import Runlength
import Dct

data Image = RGB [[RGBPixel]] | YRB [[YRBPixel]]

data RGBPixel = RGBPixel
  {
    red   :: Double,
    green :: Double,
    blue  :: Double
  } deriving (Show)


data YRBPixel = YRBPixel
  {
    intensity :: Double,
    hueRed    :: Double,
    hueBlue   :: Double
  } deriving (Show)


class Jpeg where

  --this jpegEncoder is all intergraton of functions listed below.
  --it will change from image to byte string
  --(for visualization of final representation, output is char)
  jpegEncoder :: Image -> String

  --First step is changing from rgb to yrb
  --image data type has three channels
  rgbToyrb :: Image -> Image
  
  --pixel has just three double values
  --data type is going to be changed from RGB to YRB
  rgbToyrb':: RGBPixel -> YRBPixel
  
  --downsampling is reducing resolution.
  --Normally ration of resolution -> y(intensity) : hueRed : hueBlue
  --is going to be 4:2:2, but in this case,
  --4:4:4 which is often used indigital camera
  downSampling :: Image -> [[[Double]]]
  
  --In the process of downsampliing, data type is changed to list of double
  yrbToDouble :: YRBPixel -> [Double]
  
  --next step of downsampling is block spliting which each channel is truncated with 8 by 8
  blockSpliting :: [[[Double]]] -> [[[[Double]]]]
  
  --next step after dct is quantization in each pixel.
  --As a matter of fuct, its process is just divided by values of quantization tables
  --in a floating manner.
  --values of elements on quantization table is set by Jpeg standard.
  
  quantization :: [[Double]] -> [[Double]] -> [[Int]]
  quantizationTable :: [[Double]]
  
  --next step after quantization is jigzag scan which traverse two dimensional tables in a jigzag way.
  --It will let 2D tables 1D
  jigzag  :: [[Int]] -> [Int]
  jigzag' :: Int -> Int -> [[Int]] -> [Int]
  
  --next step is seperating DC component and AC component.
  -- in 64 elements of block, a first value denotes DC component.
  -- and rest of them is AC component
  dcComponent :: [[[Int]]] ->  [[Int]]
  acComponent :: [[[Int]]] -> [[[Int]]]

  --For DC component, the difference of previous values over multiple blocks are stored,
  --except values of first block in each channel.
  takeDifference  :: [Int] ->  [Int]
  takeDifference' :: [Int] ->  [Int]

  --For AC component, things are more complicated..
  --All of non-zero code will turn into
  
  --[((A,B),C)]
  --A = number of zero before this value (4 bit fixed)
  --B = minimum bits eough to express coefficient value (4 bit fixed)
  --C = its value
  zeroLength  :: [Int] -> [((Int,Int),Int)]
  zeroLength' :: [Int] -> Int -> [((Int,Int),Int)]
  
  --integer is going to be transformed into bits in this case sequence of char type
  --first argument is expected length of bits 
  bitsTransform  :: Int -> Int -> String
  bitsTransform' :: Int -> String

  --this is sub component of bits transform
  zeroPrepare :: Int -> String
  
  --this is a tool for feeding AC component into bitsTransform function.
  acBits :: ((Int,Int),Int) -> String
  
instance Jpeg where

  --this is all of process when image is encoded as .jpg or .jpeg
  jpegEncoder a = dc ++ ac
     where b  = map (map jigzag) $ map (map (quantization quantizationTable)) $ map (map dct2D) $ blockSpliting $ downSampling $ rgbToyrb a
           dc = concatMap (concatMap (bitsTransform 8)) $ map (takeDifference) $ dcComponent b
           ac = concatMap (concatMap (concatMap acBits)) $ map (map zeroLength) $ acComponent b
           
           
  --each 2D picture is converted to YRB
  rgbToyrb (RGB a) = YRB (map (map (rgbToyrb')) a)
  
  --actual operation as a pixel level is done here.
  --this transformation is defined by Jpeg.
  rgbToyrb' a = YRBPixel {intensity = i,hueRed = hr,hueBlue = hb}
     where i  =   0.299 *(red a) + 0.587 *(green a) + 0.114 *(blue a) - 128
           hr = - 0.1687*(red a) - 0.3313*(green a) + 0.5  * (blue a) + 128
           hb =   0.5   *(red a) - 0.4187*(green a) - 0.0813*(blue a) + 128
           
           
  --scaling down resolution of Hue-red and Hue-green with respect to intensity
  --in this case, just chaging data type from my own "YRB" to Double
  downSampling (YRB a) = map (map (yrbToDouble)) a
  
  --this is type conversion                        
  yrbToDouble a = [i,hr,hb]
     where i  = intensity a
           hr = hueRed  a 
           hb = hueBlue a
           
  --block spliting is spliting whole resolutions into 8 by 8 blocks
  blockSpliting a = map (map (group 8)) a
  
  --quantization is dividing by values of quantization tables and converting integer.
  quantization = map2 (map2 (((round .) . (/)) ))
  
  --quantization is defined by Jpeg standard
  quantizationTable =
    [
      [16,11,10,16,24,40,51,61],
      [12,12,14,19,26,58,60,55],
      [14,13,16,24,40,57,60,56],
      [14,17,22,29,51,87,80,62],
      [18,22,37,56,68,109,103,77],
      [24,35,55,64,81,104,113,92],
      [49,64,78,87,103,121,120,101],
      [72,92,95,98,112,100,103,99]
    ]  

  --two dimensional blocks are converted into 1d 64 elements list
  jigzag a = jigzag' ((length a)*2) 0 a
  
  --jigzag process has 3 arugments
  --argument is (depth of jigzag(8by8->16),current depth,2D list).
  jigzag' m a b
  -- current depth is the end of jigzag, prepare empty list
    | m == a    = []
  -- if the current depth is even, append the first element of extracted list as usual
    | even a    = map (head) c ++ r
  -- if the current depth is odd, append as reversed to let the traversal jigzag
    | otherwise = (reverse $ map (head) c) ++ r
  -- after half step of jigzag, it has empty list which needs to be excluded
    where b' = (zip [0..] . filter (/=[])) b
  -- according to the depth, lists are extracted
          c = (snd . unzip . filter (\(x,y)->x<a))  b'
  -- none-extracted lists is going to brought as next arguments
          d = (snd . unzip . filter (\(x,y)->x>=a))  b'
          r = jigzag' m (a+1) ((map (tail) c) ++ d)
          
          
  --after jigzag scan, DC & AC component is divided.
  dcComponent = map (map (head))
  acComponent = map (map (tail))

  --DC component is encoded by the value of its difference.
  takeDifference (h:t) = h : takeDifference' t
  takeDifference' [h]   = []
  takeDifference' (h:t) = h - (head t) : takeDifference' t
  
  --AC component is only non-zero component is stored unless it has
  --sequence more than 15 zero.
  
  zeroLength a = zeroLength' a 0

  --
  zeroLength' [] _ = []
  --1st argument is encoded list, second argument is variables to store previous successive number of 0
  zeroLength' (h:t) s
  --if it comes zero, you will jump it storing number of zero 
    | h == 0    = zeroLength' t (s+1)
  --if it comes non zero, you will store it with numbers of minimum bits for this value
    | otherwise = ((s,bit+1),h) : zeroLength' t 0
  --minimum bits is calculated as following.
    where bit = round $ logBase 2 $ fromIntegral (h::Int)

  --integer turns into bits but as a string format
  --1st argument is length of bits showing discrimination for decoding
  bitsTransform a b = (zeroPrepare (a-l)) ++ c
    where c = (reverse . bitsTransform') b
          l = length c
          
  --int -> bit are done by dividing by 2 repeatedly.
  --if it comes zero, it will be ended
  bitsTransform' 0 = []
  bitsTransform' b
  --if mode b 2 == 0, then, encode as 0
    | mod b 2 == 0 = "0" ++ (bitsTransform' (b `div` 2))
  --if mode b 2 != 0, then, encode as 1
    | otherwise    = "1" ++ (bitsTransform' (b `div` 2))

  --if there is more space to be filled in, insert "0"
  zeroPrepare 0 = ""
  zeroPrepare a = "0" ++ (zeroPrepare (a-1))
  
  --AC component is fed into transform and concatenate.
  acBits ((a,b),c) = (bitsTransform 4 a) ++ (bitsTransform 4 b) ++ (bitsTransform b c)


main = do
  print "hi"

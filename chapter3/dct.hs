{-# LANGUAGE MultiParamTypeClasses #-}

module Dct
  (
    DCT(..)
    
  ) where


--import Data.Complex

import MyLib.General
import MyLib.Matrix
import MyLib.MyMath
import MyLib.MyComplex

import Runlength


class DCT where
  
  dct1D :: [Double] -> [[Double]]
  
  dct2D :: [[Double]] -> [[Double]]
  
  dct2D'  :: [[Double]] -> [[Double]]
  dct2D'' :: [[Double]] -> [[Double]]
  
  base' :: (Double -> Double) -> Double -> [[Double]]
  
instance DCT where
  
  dct1D a = dot' b c
     where b = atLeast2d a
           l = fromIntegral (length a) / 1.0
           c = base' cos l

  dct2D = dct2D'' . dct2D'
          
  dct2D' a = dot' a c
     where l = fromIntegral (length a) / 1.0
           c = base' cos l
           
           
  dct2D'' a  = dot' b c
     where l = fromIntegral (length a) / 1.0
           b = t' a 
           c = base' cos l
  
           
  --base function generation
  --sub function of base generator
  base' f a = map (\y -> map (\x -> f ( 2 * pi * x * y / a )) [0..b] ) [0..b] where b = a - 1
  
  
--getL :: Double -> [Double]
{--
getR  :: Double -> [Double]
getR a = map (\x -> (2*pi) / a * x) [1..a]

sinList :: Double -> [Double]
sinList = map (sin) . getR

cosList :: Double -> [Double]
cosList = map (cos) . getR

listCopy  :: Int -> [a] -> [[a]]
listCopy 0 _ = []
listCopy a b = b : listCopy (a-1) b
--}

  
          
main = do
  
  --let a = map (*128) $ sinList 16
  
  --let c = dct2D b
  --let d = quantization c quantizationTable
  
  --let q = quantizationTable
  --print d
  let s = [
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,60,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
        ]
        
  
  print "h"
  
  

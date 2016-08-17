{-# LANGUAGE MultiParamTypeClasses #-}

module Convolve
  (
    Convolution(..),
    Kernel1D(..),
    Kernel2D(..)
  ) where


--import Data.Complex

import MyLib.General
import MyLib.Matrix
import MyLib.MyMath
import MyLib.MyComplex

import Runlength

type Kernel1D = [Double]

type Kernel2D = [[Double]]


class Convolution where
  
  --1 dimensional convolution
  convolve1D :: [Double] -> Kernel1D -> [Double]
  
  --apply boundry compensation to the top of list and the bottom of list
  --iterate 1d list
  boundry1D :: Int -> [Double] -> [Double]
  
  --take first element and insert on top of the list
  --iterate 1d list
  filling1D  :: Int -> [Double] -> [Double]
  
  --this function is just insertion N value on top of list
  -- N is first argument
  filling'   :: Int -> Double -> [Double]
  
  --actual operation of convolution
  convolve1D' :: [Double] -> Kernel1D -> [Double]
  
  
  --2 dimensional convolution
  convolve2D  :: [[Double]] -> Kernel2D -> [[Double]]
  
  --apply boundry compensation to the top of list and the bottom of list
  --iterate 2d list
  boundry2D :: Int -> [[Double]] -> [[Double]]
  
  --take first element and insert on top of the list
  --iterate 2d list 
  filling2D  :: Int -> [[Double]] -> [[Double]]
  
  --actual operation of 2d convolution
  convolve2D'  :: [[Double]] -> Kernel2D -> [[Double]]
  convolve2D'' :: [[Double]] -> Kernel2D ->  [Double]
  
  
instance Convolution where
  
  convolve1D d k = convolve1D' dd k
  -- get the half of kernel size to insert boundry compensation for convolution beforehand
    where s = (length k) `div` 2
  -- apply boundry compensation
          dd = boundry1D s d
          
          
  boundry1D s a = dt
  --boundry compensation here is taking nearest elements on the list,
  --and insert it to the top and bottom of the list.
    where r  = (reverse . filling1D s . reverse) a
          dt = filling1D s r
          
          
  filling1D s a = b
  --take first element and insert on top of given list
    where rh = filling' s (head a)
          b  = (++) rh a
          
          
  convolve1D' d k
  --if the size of remained vector is less than kernel size, stop convolution.
    | lk > ld   = []
  --convolution is done by taking first N element from the list
  --and multiply with kernel and then sum up
  --first N element is kernel size
    | otherwise = foldl (+) 0 ( map2 (*) k (take' lk d)) : (convolve1D' (tail d) k)
    where ld = length d
          lk = length k
          
          
  convolve2D d k = convolve2D' dd k
  --for 2d convolution, boundry compensation have to be done twice for rows and columns.
    where s  = (length k) `div` 2
          dd = (boundry2D s . boundry2D s) d

          
  boundry2D s a = dt
  --boundry2d is identical to boundry 1d except input list is 2d
    where r  = (map (reverse) . filling2D s . map (reverse)) a
          dt = (t' . filling2D s) r
          
          
  filling2D s a = b
  --filling1d is identical to filling 1d except input list is 2d
    where rh = map (\x -> filling' s (head x)) a
          b  = map2 (++) rh a
          
  --according to the kernel size, you fill up nearest values as window.
  filling' 0 _ = []
  filling' i a = a : filling' (i-1) a
  
  --this operation is to iterate different rows
  convolve2D' d k
  --if number of rest element is less than kernel size, stop
    | lk > ld   = []
  --apply column convolution to the each list and append them
    | otherwise = (convolve2D'' d k) : (convolve2D' (tail d) k)
    where lk = length k
          ld = length d

  --this operation is to iterate different columns
  convolve2D'' d k
  --if number of rest element is less than kernel size, stop
    | lk > ld   = []
  --actual convolution operation.
  --two dimensional list is concatenated to 1d and then, elementwise multiplication is done.
  --finally, sums up everything.
    | otherwise = foldl (+) 0 ( map2 (*) (concat k) (concatMap (take' lk) d) ) :(convolve2D'' (map (tail) d) k)
    where lk = length $ head k
          ld = length $ head d
          
          
          
main = do
  
  print $ convolve1D  [1..10] [1,2,3,2,-1]
  
  print $ convolve2D [[1..5],[0..4],[-1..3],[1..5],[1..5]] [[1..3],[1..3],[1..3]]
  
  --print $ convolve2D [[1..3],[1..3],[1..3]] [[1..5],[0..4],[-1..3],[1..5],[1..5]]
  
  --print $ filling 4 2
  
  print "hi"





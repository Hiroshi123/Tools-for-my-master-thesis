{-# LANGUAGE MultiParamTypeClasses #-}

module MyLib.MyMath
  (
    MyMath(..)
  ) where

import MyLib.General

class MyMath where
  
  --difinition of triangle function 
  cos' :: Double -> Double
  sin' :: Double -> Double
  
  --pow {two arguments} (e.g. 2^3 -> 8)
  pow'  :: Double -> Int -> Double

  --element to list (e.g. eToList 2 3 -> (2,2,2) )
  eToList :: a -> Int -> [a]
  
  --get even number
  evenL :: Int -> [Int]
  
  --get odd number
  oddL  :: Int -> [Int]
  
  --element to list (e.g. eToList 2 3 -> (2,2,2) )
  evenLIndex :: [a] -> [a]
  oddLIndex  :: [a] -> [a]
  

instance MyMath where

  --this is pow!
  pow' = (foldl (*) 1 .) . eToList
  
  --make list whose all elements are all 1st arguments, and its length is 2nd argument
  --for instance eToList 3 5 -> [3,3,3,3,3]
  
  eToList _ 0 = []
  eToList a b = a : eToList a (b-1)
  
  --get successive even number from 2
  evenL a = filter (even) [1..b] where b = 2 * a

  --get successive even number from 2
  oddL  a = filter (odd)  [1..b] where b = 2 * a
  
  --retrieve all elements whose index is even
  evenLIndex = ((snd ). unzip) . filter (\(x,y) -> even x) . zip [1..]
  --retrieve all elements whose index is odd
  oddLIndex  = ((snd ). unzip) . filter (\(x,y) -> odd  x) . zip [1..]
  
  
  --oddLIndex a = snd $ unzip $ filter (\(x,y) -> odd x) $ zip [1..] a
  
  --sin is defined by cos
  sin' a = cos' (pi / 2.0 - a)
  
  --get cos
  --1. prepare denominator terms for macroline expansion which is pow of entering angle
  --2. prepare numerator terms for macroline expansion which is fractional of entering even number 
  --3. devide output of 1 by 2
  --4. elements whose index is even has negative, whose index is odd has positive
  --5. sum up all of terms
  cos' a = foldl (+) (foldl (-) 0 e) f
    where b = (oddLIndex . scanl (*) 1 . eToList a) 100
          c = oddLIndex . scanl (*) 1 $ [1..100]
          d = map2 (/) b c
          e = evenLIndex d
          f = oddLIndex  d




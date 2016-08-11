{-# LANGUAGE MultiParamTypeClasses #-}


module MyLib.Matrix
  (
    MyMatrix(..)
  ) where


import MyLib.General

main = do
  print "hi"

class MyMatrix where
  
  --inner product
  dot' :: (Num a) => [[a]] -> [[a]] -> [[a]]
  
  --transpose
  t' :: [[a]] -> [[a]]

  atLeast2d :: [a] -> [[a]]
  

instance MyMatrix where

  --inner product
  dot' a b = group (length a) ( map ( foldl (+) 0 ) [ (map2 (*) x y)  | x <- a, y <- t])
     where t = t' b

  atLeast2d a = [a]
  
  --transpose
  t' ([]:_) = []
  t' x = (map head x) : t' (map tail x)
  


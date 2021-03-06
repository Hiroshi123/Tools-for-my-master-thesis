{-# LANGUAGE MultiParamTypeClasses #-}

module MyLib.MyComplex
  (
    MyComplex(..),
    Complex'(..)
  ) where

import Control.Monad
import MyLib.Matrix
import MyLib.General

data Complex' = Complex' (Double,Double) deriving (Show)


class MyComplex where
  
  --basic operation on complex space
  iAdd :: Complex' -> Complex' -> Complex'
  iSub :: Complex' -> Complex' -> Complex'
  iMul :: Complex' -> Complex' -> Complex'
  iDiv :: Complex' -> Complex' -> Complex'
  
  iDot :: [[Complex']] -> [[Complex']] -> [[Complex']]
  
  --change real to complex
  toComplex  :: Double -> Complex'
  
  --change real to complex
  toReal  :: Complex' -> Double
  
  --get conjuguate
  conjuguate :: Complex' -> Complex'
  
  
instance MyComplex where

  --basic operation on complex space
  iAdd (Complex' (k1,v1)) (Complex' (k2,v2)) = Complex' (k1+k2,v1+v2)
  iSub (Complex' (k1,v1)) (Complex' (k2,v2)) = Complex' (k1-k2,v1-v2)
  iMul (Complex' (k1,v1)) (Complex' (k2,v2)) = Complex' (k1*k2-v1*v2,k1*v2+k2*v1)
  iDiv (Complex' (k1,v1)) (Complex' (k2,v2)) = Complex' (k1/k2,v1/v2)
  
  --inner product
  iDot a b = group (length a) ( map ( foldl (iAdd) (Complex' (0,0)) ) [ (map2 (iMul) x y)  | x <- a, y <- t])
     where t = t' b
           
                   
           
  --change real to complex
  --toComplex (Left (Complex' (a,b))) = Complex' (a,b)
  --toComplex (Complex' (a,b))) = Complex' (a,0)
  
  
  toComplex a = Complex' (a,0)
  
  
  --change complex to real( just discard imaginary part)
  toReal (Complex'(a,b)) = a
  
  --take conjuguate
  conjuguate (Complex'(a,b)) = Complex' (a,-b)
  
  

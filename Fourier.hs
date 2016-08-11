{-# LANGUAGE MultiParamTypeClasses #-}


--import Data.Complex

import MyLib.General
import MyLib.Matrix
import MyLib.MyMath
import MyLib.MyComplex


class Fourier where
  
  --Fourier transformation
  --Input  : transformed function can be defined by real number
  --Output : two dimensional complex number
  -- (why two dimensional? Answer : matrix can be more easily treated than vector.) 
  transform :: [Double] -> [[Complex']]

  
  --Inverse Fourier transformation
  --Input  : Coefficients of Fourier Series
  --Output : one dimensional real number
  
  invTransform :: [[Complex']] -> [Double]

  
  --base function preparation
  base  :: Double -> [[Complex']]
  
  --sub function of base generator 
  base' :: (Double -> Double) -> Double -> [[Double]]
  
  
instance Fourier where
  
  --Fourier transformation
  --Fourier transformation is inner product of complex number
  --1. Let input complex data type
  --2. Prepare base function accroding to sampling frequency; namely, length of input.
  --(Do not forget to take conjuguate; sin has negative from its definition.)
  --3. Compute inner dot between input and base function
  transform a = iDot b c
     where b = (atLeast2d . map (toComplex)) a
           l = fromIntegral (length a) / 1.0
           c = (map (map (conjuguate)) . base) l
           
           
  --inverse Fourier transformation
  --inverse Fourier transformation is almost identical with Fourier transformation
  --Note
  -- # input needs to be transposed after Fourier transformation.
  -- # Base do not need to take conjuguate from its definition.
  -- # Do not forget multiply devide by length of input in the end
  
  invTransform a = concatMap (map (\x -> (toReal . iDiv x) (Complex' (l,1)) )) (iDot b c) 
     where b = t' a
           l = fromIntegral (length a) / 1.0
           c = base l
           
           
  --base function preparation
  base  a  = map (\s -> map (\(x,y) -> Complex' (x,y)) s ) d
    where b = base' cos' a
          c = base' sin a
          d = map2 (zip) b c
          
          
  --base function generation
  --sub function of base generator
  base' f a = map (\y -> map (\x -> f ( 2 * pi * x * y / a )) [0..b] ) [0..b] where b = a - 1
  
main = do
  
  let o = [1,-3.2,3,2,1,-2]
  
  print "-----------------"
  
  let out = transform o
  print out
  
  let back = invTransform out
  print back
  
  
  --let s = 3 :+ (-4)
  --let v = 3 :+ 4
  --print $ s * v
  
  --print $ realPart s
  --print $ imagPart s
  --let s = !2 :+ !4
  
  
  print "hei"
  

  

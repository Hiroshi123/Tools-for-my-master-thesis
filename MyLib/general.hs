{-# LANGUAGE MultiParamTypeClasses #-}

module MyLib.General
  (
    General(..)
  ) where



class General where
  map2 :: (a -> a -> b) -> [a] -> [a] -> [b]
  
  --map2' :: ([a] -> [a] -> a) -> [a] -> [a] -> [a]
  
  take'  :: Int -> [a] -> [a]
  take'' :: Int -> [a] -> [a]
  
  drop' :: Int -> [a] -> [a]
  
  group :: Int -> [a] -> [[a]]

instance General where
  map2 _  a []= []
  map2 _ [] a = []
  
  map2 f (h1:t1) (h2:t2) = f h1 h2 : map2 f t1 t2
  
  take' a [] = []
  take' 0 a  = []
  take' i (h:t) = h : (take' (i-1) t)
  
  take'' a l = reverse $ foldl (\x y -> if length x < 2 then y : x else x) [] l

  drop' 0 a  = a
  drop' i (h:t) = drop' (i-1) t

  group  _ [] = []
  group i t = take' i t : group i (drop' i t)
  

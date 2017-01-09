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
  
  --this group needs two arguments, truncate input list into every designated index
  group :: Int -> [a] -> [[a]]
  
  --this group needs one argument, let successive symbols intergrate.
  --e.g.."sssaaaavvvxxx"-> " "sss","aaaa","vvv","xxx" "
  group' :: (Eq a) => [a] -> [[a]]

  --this fucntion returns first N successive symbols.
  --e.g.."sssaaaavvvxxx"-> "sss"
  accord' :: (Eq a) => [a] -> [a]
  
  
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
  
  group' [] = []
  group' h = (take l h) : group' (drop l h)
     where s = accord' h
           l = length s
           
           
  --intergration of successive symbol
  accord' [a] = [a] -- if the symbol is just alone, return it
  accord' (h:t)
    | h == head t = [h] ++ (accord' t)
    | otherwise   = [h]
    -- if the next symbol is not equal to current symbol, return it.

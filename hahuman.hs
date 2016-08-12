{-# LANGUAGE MultiParamTypeClasses #-}


import MyLib.LeafTree
import MyLib.TreeHash


class Hahuman where

  --this is integration of all code.
  hahuman :: (Ord a) => [a] -> (String,[(a,String)])
  
  --make dictionary for encoding 
  makeDict  :: LeafTree k v -> [(k,String)]
  makeDict' :: String -> LeafTree k v -> [(k,String)]
  
  --1st arugment: signal
  --2nd arugment: tree which has encode information
  encode  :: (Ord k) => [k] -> LeafTree k Int -> String
  encode' :: (Ord k) => String -> LeafTree k v -> k -> String
  
  
instance Hahuman where
  
  --huhman coding integration code.
  --1 : cumulativeFromList ( TreeMap )
  --get cumulative frequency given input list in a tree structure,
  --2 : treeToList
  --change the tree into list,
  --3 : insertsVL ( LeafTree )
  --again change to tree but in this time, leaf tree
  -- leaf tree are allowed us to put anything on their intermidiate node.
  --4. create dictionary
  --4. encode given created leaf tree
  
  hahuman l = (encode l a, makeDict a)
    where a = ((insertsVL . treeToList) . cumulativeFromList) l
    
    
  makeDict = makeDict' ""
  makeDict' _ None = []
  makeDict' s (LNode (Leaf k v) r) = (k,t) : makeDict' u r
    where t = s ++ "0"
          u = s ++ "1"
          
          
  --encode' _ None = []
  encode l tree = foldr (\x y -> (encode' "" tree x) ++ y) "" l
  encode' s (LNode (Leaf k v) r) e
    | e == k    = t
    | otherwise = encode' u r e
    where t = s ++ "0"
          u = s ++ "1"
          
          
main = do
  
  let input = "abacbacccdbbb"
  
  print $ hahuman input
  
  print "hei"
  
  
  

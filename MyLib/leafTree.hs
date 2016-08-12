{-# LANGUAGE MultiParamTypeClasses #-}


module MyLib.LeafTree(
  LeafTree(..),
  FuncForLeafTree(..)
  ) where


--LeafTree is only leaves are able to hold pairs of key and value.
--Note LNode has just two argument unlike Node which is defined inside of TreeMap
data LeafTree k v = None | Leaf k v | LNode (LeafTree k v) (LeafTree k v) deriving (Show)

    
    
--This is class for functions which is for LeafTree
class FuncForLeafTree where
  --insert elements sorting by values
  insertsVL :: Ord v => [(k,v)] -> LeafTree k v
  insertVL  :: Ord v => k -> v -> LeafTree k v -> LeafTree k v
  
  
instance FuncForLeafTree where

  --this is just looping list of elements
  insertsVL = foldl (\a (k, v) -> insertVL k v a) None
  
  --this creates left-shifted tree
  insertVL k v None = LNode (Leaf k v) None
  insertVL k v (LNode (Leaf lk lv) r)
    | v > lv    = LNode (Leaf k v) (LNode (Leaf lk lv) r)
    | otherwise = (LNode (Leaf lk lv) (insertVL k v r))

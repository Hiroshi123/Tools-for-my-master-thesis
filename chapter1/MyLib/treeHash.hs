{-# LANGUAGE MultiParamTypeClasses #-}


module MyLib.TreeHash
  (
    TreeMap(..),
    TreeHash(..)
  ) where

--There are two different tree. TreeMap is ordinary tree where you can assign key and value
--without any distinct of node and leaf.
data TreeMap k v  = Nil | Node k v (TreeMap k v) (TreeMap k v) deriving (Show)

class TreeHash where
  
  --find node on the tree
  search :: Ord k => k -> TreeMap k v -> Maybe v
  
  --change list to tree hash
  fromList :: Ord k => [(k, v)] -> TreeMap k v

  --this is leaf
  singleton :: k -> v -> TreeMap k v

  --insert sort by key
  insertK :: Ord k => k -> v -> TreeMap k v -> TreeMap k v
  --insert sort by value
  insertV :: Ord v => k -> v -> TreeMap k v -> TreeMap k v
  
  -- frequency table
  -- iterating list
  freqTableFromList :: Ord k => [k] -> TreeMap k Int
  
  --adding elements given one query
  freqTable :: Ord k => k -> TreeMap k Int -> TreeMap k Int
  
  --sort by value
  --sortByV :: TreeMap k Int -> TreeMap k Int
  treeToList :: TreeMap k v -> [(k, v)]
  
  --traverse
  --compareL :: TreeMap k v -> (k,v)
  
instance TreeHash where
  
  fromList = foldl (\a (k, v) -> insertK k v a) Nil

  --if this node is leaf 
  singleton k v = Node k v Nil Nil
  --when you insert pairs of key & value, this will sort elements by keys
  insertK x y Nil = singleton x y
  insertK x y (Node k v l r)
    | x == k    = Node x y l r
    | x < k     = Node k v (insertK x y l) r
    | otherwise = Node k v l (insertK x y r)
    
  --when you insert pairs of key & value, this will sort elements by values
  insertV x y Nil = singleton x y
  insertV x y (Node k v l r)
    | y == v    = Node x y l r
    | y < v     = Node k v (insertV x y l) r
    | otherwise = Node k v l (insertV x y r)
    
    
  -- input  : list 
  -- output : Tree 
  freqTableFromList = foldl (\x y -> freqTable y x) Nil
  
  -- if there are no elements at the point you come to a leaf, add new node
  freqTable a  Nil = Node a 1 Nil Nil
  -- if there are same elements in the list, add +1 on top of its index
  freqTable a (Node k v l r)
    | a == k    = Node k (v+1) l r
    | a < k     = Node k v (freqTable a l) r
    | otherwise = Node k v l (freqTable a r)
    -- search by key, go depth guided by key

  --this is search function
  search _ Nil = Nothing
  search x (Node k v l r)
    | x == k    = Just v
    | x < k     = search x l
    | otherwise = search x r
    
  --change tree to list
  treeToList tree = traverse tree [] where
    traverse Nil xs = xs
    traverse (Node k v l r) xs = traverse l ((k, v) : traverse r xs)
    



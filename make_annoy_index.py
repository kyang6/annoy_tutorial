'''
Usage: python2 make_annoy_index.py \
    --embeddings=<embedding path> \
    --num_trees=<int> \
    --verbose

Generate an Annoy index and lmdb map given an embedding file

Embedding file can be
  1. A .bin file that is compatible with word2vec binary formats.
     There are pre-trained vectors to download at https://code.google.com/p/word2vec/
  2. A .gz file with the GloVe format (item then a list of floats in plaintext)
  3. A plain text file with the same format as above

'''

import annoy
import lmdb
import os
import sys
import argparse

from vector_utils import get_vectors

'''
private function _create_args()
-------------------------------
Creates an argeparse object for CLI for create_index() function

Input:
    Void

Return:
    args object with required arguments for threshold_image() function

'''
def _create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", help="filename of the embeddings", type=str)
    parser.add_argument("--num_trees", help="number of trees to build index with", type=int)
    parser.add_argument("--verbose", help="print logging", action="store_true")

    args = parser.parse_args()
    return args


'''
function create_index(fn, num_trees=30, verbose=False)
-------------------------------
Creates an Annoy index and lmdb map given an embedding file fn

Input:
    fn              - filename of the embedding file
    num_trees       - number of trees to build Annoy index with
    verbose         - log status

Return:
    Void
'''
def create_index(fn, num_trees=30, verbose=False):
    fn_annoy = fn + '.annoy'
    fn_lmdb = fn + '.lmdb' # stores word <-> id mapping

    word, vec = get_vectors(fn).next()
    size = len(vec)
    if verbose:
        print("Vector size: {}".format(size))
    
    env = lmdb.open(fn_lmdb, map_size=int(1e9))
    if not os.path.exists(fn_annoy) or not os.path.exists(fn_lmdb):
        i = 0
        a = annoy.AnnoyIndex(size)
        with env.begin(write=True) as txn:
            for word, vec in get_vectors(fn):
                a.add_item(i, vec)
                id = 'i%d' % i
                word = 'w' + word
                txn.put(id, word)
                txn.put(word, id)
                i += 1
                if verbose:
                    if i % 1000 == 0:
                        print(i, '...')
        if verbose:
            print("Starting to build")
        a.build(num_trees)
        if verbose:
            print("Finished building")
        a.save(fn_annoy)
        if verbose:
            print("Annoy index saved to: {}".format(fn_annoy))
            print("lmdb map saved to: {}".format(fn_lmdb))
    else:
        print("Annoy index and lmdb map already in path")


if __name__ == '__main__':
    args = _create_args()
    create_index(args.embeddings, num_trees=args.num_trees, verbose=args.verbose)





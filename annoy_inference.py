'''
Usage: python2 annoy_inference.py \
    --token='hello' \
    --num_results=<int> \
    --verbose

Query an Annoy index to find approximate nearest neighbors

'''
import annoy
import lmdb
import argparse


FN_ANNOY = 'glove.6B.50d.txt.annoy'
FN_LMDB = 'glove.6B.50d.txt.lmdb'
VEC_LENGTH = 50


a = annoy.AnnoyIndex(VEC_LENGTH)
a.load(FN_ANNOY)
env = lmdb.open(FN_LMDB, map_size=int(1e9))


'''
private function _create_args()
-------------------------------
Creates an argeparse object for CLI for calculate() function

Input:
    Void

Return:
    args object with required arguments for threshold_image() function

'''
def _create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="query word", type=str)
    parser.add_argument("--num_results", help="number of results to return", type=int)
    parser.add_argument("--verbose", help="print logging", action="store_true")

    args = parser.parse_args()
    return args

'''
private function calculate(query, num_results)
-------------------------------
Queries a given Annoy index and lmdb map for num_results nearest neighbors

Input:
    query           - query to be searched
    num_results     - the number of results

Return:
    ret_keys        - list of num_results nearest neighbors keys

'''
def calculate(query, num_results, verbose=False):
    ret_keys = []
    with env.begin() as txn:
        id = int(txn.get('w' + query)[1:])
        if verbose:
            print("Query: {}, with id: {}".format(query, id))
        v = a.get_item_vector(id)
        for id in a.get_nns_by_vector(v, num_results):
            key = txn.get('i%d' % id)[1:]
            ret_keys.append(key)
    if verbose:
        print("Found: {} results".format(len(ret_keys)))
    return ret_keys


if __name__ == '__main__':
    args = _create_args()
    print(calculate(args.token, args.num_results, args.verbose))














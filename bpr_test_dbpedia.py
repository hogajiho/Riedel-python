from bpr_n import BPR
import dbpedia_parser as dp
from numpy import mean

if __name__ == "__main__":
	etd, rd = dp.parse_and_return_dict()
	freq_relations, freq_tuples = dp.frequent_dict(etd, rd)
	print len(freq_relations)
	train_set, test_set = dp.make_train_test_set(etd, rd, freq_relations, freq_tuples)
	n_tup = len(freq_tuples)
	n_rel = len(freq_relations)
	print len(train_set), len(test_set)


	exp_num = 1
	f_score = list()
	n_score = list()
	nf_score = list()

	# Dimension 30 Matrix Factorization
	print "Training f model..."
	for i in range(exp_num):
		bpr = BPR(50, n_tup, n_rel, "f")
		bpr.train(train_set)
		#print "Testing f model..."
		f_score.append(bpr.test(test_set))

	# Neighborhood model
	print "Training n model..."
	for i in range(exp_num):
		bpr = BPR(50, n_tup, n_rel, "n")
		bpr.train(train_set)
		#print "Testing n model..."
		n_score.append(bpr.test(test_set))


	print "Training nf model..."
	for i in range(exp_num):
		bpr = BPR(50, n_tup, n_rel, "n")
		bpr.train(train_set)
		#print "Testing nf model..."
		nf_score.append(bpr.test(test_set))

	print mean(f_score)
	print mean(n_score)
	print mean(nf_score)
from collections import defaultdict
import codecs


def parse_and_return_dict():
	f = codecs.open("korean-dbpedia", "r", encoding="utf-8")
	entity_tuple_dict = defaultdict(list)
	relation_dict = defaultdict(list)

	for line in f.readlines():
		line = line.split("\t")
		line[2] = line[2][:-1]
		entity_tuple = (line[0], line[2])
		relation = line[1]

		entity_tuple_dict[entity_tuple].append(relation)
		relation_dict[relation].append(entity_tuple)

	f.close()

	return entity_tuple_dict, relation_dict


def frequent_dict(etd, rd):

	cnt = 0
	freq_relations = set()
	for key in rd.keys():
		if len(rd[key]) > 20:
			cnt += 1
			freq_relations.add(key)

	freq_relations = set(list(freq_relations)[:500])

	print cnt

	for key in etd.keys():
		etd[key] = [x for x in etd[key] if x in freq_relations]

	cnt = 0
	freq_tuples = set()
	for key in etd.keys():
		if len(etd[key]) > 2:
			cnt += 1
			freq_tuples.add(key)

	print cnt

	return list(freq_relations), list(freq_tuples)

def make_train_test_set(etd, rd, freq_relations, freq_tuples):
	f1 = codecs.open("freq_relations", "w", encoding="utf-8")
	f2 = codecs.open("freq_tuples", "w", encoding="utf-8")
	train_set = list()
	test_set = list()

	for i in range(len(freq_relations)):
		f1.write(freq_relations[i])
		f1.write("\t")
		f1.write(str(i))
		f1.write("\n")

	for i in range(len(freq_tuples)):
		f2.write(freq_tuples[i][0])
		f2.write("\t")
		f2.write(freq_tuples[i][1])
		f2.write("\t")
		f2.write(str(i))
		f2.write("\n")

	f1.close()
	f2.close()

	for i in range(len(freq_tuples)):
		relation_list = [x for x in etd[freq_tuples[i]] if x in freq_relations]
		for j in range(len(relation_list)-1):
			train_set.append((i, freq_relations.index(relation_list[j])))
		test_set.append((i, freq_relations.index(relation_list[-1])))

	return train_set, test_set


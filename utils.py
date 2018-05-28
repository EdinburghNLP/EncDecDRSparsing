# -*- coding: utf-8 -*-
import torch
def readfile(filename):
	data = []
	with open(filename, "r") as r:
		while True:
			l1 = r.readline().strip()
			l2 = r.readline().strip()
			l3 = r.readline().strip()
			if l1 == "":
				break
			data.append((l1.split(), [ w.lower() for w in l1.split()], l2.split(), l3.split()))
	return data

def readpretrain(filename):
	data = []
	with open(filename, "r") as r:
		while True:
			l = r.readline().strip()
			if l == "":
				break
			data.append(l.split())
	return data

def get_from_ix(w, to_ix, unk):
	if w in to_ix:
		return to_ix[w]

	assert unk != -1, "no unk supported"
	return unk

def data2instance(trn_data, ixes):
	instances = []
	for one in trn_data:
		instances.append([])
		## words
		instances[-1].append(torch.LongTensor([get_from_ix(w, ixes[0][0], ixes[0][1]) for w in one[0]]))
		instances[-1].append(torch.LongTensor([get_from_ix(w, ixes[1][0], ixes[1][1]) for w in one[1]]))
		instances[-1].append(torch.LongTensor([get_from_ix(w, ixes[2][0], ixes[2][1]) for w in one[2]]))

		instances[-1].append([])
		relation = 0
		for item in one[3]:
			type, idx = ixes[3].type(item)
			if type == -2:
				if idx == 4:
					if relation == 0:
						instances[-1][-1].append(idx)
					elif relation > 0:
						relation -= 1
					else:
						assert False
				elif idx >= 5 and idx <= 12:
					instances[-1][-1].append(idx)
				elif idx >= ixes[3].k_rel_start and idx < ixes[3].p_rel_start:
					instances[-1][-1].append(idx)
				elif idx >= ixes[3].p_rel_start and idx < ixes[3].k_tag_start:
					instances[-1][-1].append(idx)
				elif idx >= 13 and idx < ixes[3].k_rel_start:
					relation += 1
			else:
				relation += 1

		instances[-1].append([])
		stack = []
		pointer = 0
		#print "####"
		for item in one[3]:
			type, idx = ixes[3].type(item)
			#print "==="
			#print "stack:",stack
			#print "type, idx", type, idx
			#print "pointer",pointer
			#print "item", item
			if type == -2:
				if idx == 4:
					if stack[-1][0] >= 7 and stack[-1][0] <= 12:
						pass
					elif stack[-1][0] >= ixes[3].k_rel_start and stack[-1][0] < ixes[3].k_tag_start:
						pass
					elif stack[-1][0] == 5 or stack[-1][0] == 6:
						pass
					else:
						instances[-1][-1][stack[-2][1]].append([stack[-1][1], stack[-1][0]])
					stack.pop()
				elif idx == 5 or idx == 6:
					stack.append([idx, pointer])
					instances[-1][-1].append([])
					pointer += 1
				elif idx >= 7 and idx < ixes[3].k_tag_start:
					stack.append([idx, -2, -1])
			else:
				type = one[2].index(item[:-1])
				idx = instances[-1][2].tolist()[type] + ixes[3].tag_size
				assert type != -1 and idx != -1, "unrecogized local relation"
				stack.append([idx, type, -1])

		instances[-1].append([])
		for item in one[3]:
			type, idx = ixes[3].type(item)
			if type == -2 and idx >= ixes[3].k_tag_start and idx < ixes[3].tag_size: #variable
				pass
			elif type == -2 and (idx == 2 or idx == 3): # CARD_NUMBER and TIME_NUMBER
				pass
			else:
				if type == -2:
					instances[-1][-1].append(idx)
				else:
					type = one[2].index(item[:-1])
					idx = instances[-1][2].tolist()[type] + ixes[3].tag_size
					assert type != -1 and idx != -1, "unrecogized local relation"
					instances[-1][-1].append(idx)

		assert len(instances[-1][-1]) != 0

		instances[-1].append([])
		#print "####"
		for item in one[3]:
			type, idx = ixes[3].type(item)
			#print "==="
			#print "type, idx", type, idx
			#print "item", item
			if type == -2 and idx >= ixes[3].k_tag_start and idx < ixes[3].tag_size: #variable
				instances[-1][-1][-1].append(idx)
			elif type == -2 and (idx == 2 or idx == 3): # CARD_NUMBER and TIME_NUMBER
				instances[-1][-1][-1].append(idx)
			elif type == -1 or (type == -2 and idx >= 13 and idx < ixes[3].k_rel_start):
				if len(instances[-1][-1]) != 0:
					assert len(instances[-1][-1][-1]) <= 2 and len(instances[-1][-1][-1]) > 0
				instances[-1][-1].append([])
			#print instances[-1][-1]
	return instances

def packed_data(trn_instances, batch_size):
	packed_instances = []
	packed_instance = []
	packed_lengths = []
	packed_length = []
	packed_idx = 0
	max_length = len(trn_instances[0][0])
	for instance in trn_instances:
		if packed_idx != 0 and packed_idx % 100 == 0:
			packed_instances.append(packed_instance)
			packed_lengths.append(packed_length)
			packed_instance = []
			packed_length = []
		if len(packed_instance) == 0:
			max_length = len(instance[0])

		packed_length.append(len(instance[0]))
		instance[0] += [0 for i in range(max_length-len(instance[0]))]
		instance[1] += [0 for i in range(max_length-len(instance[1]))]
		instance[2] += [0 for i in range(max_length-len(instance[2]))]
		instance[3] += [0 for i in range(max_length-len(instance[3]))]
		packed_instance.append(instance)
		
		packed_idx += 1
	if len(packed_instance) != 0:
		packed_instances.append(packed_instance)
		packed_lengths.append(packed_length)
	return packed_instances, packed_lengths

def packed_data_orig(trn_instances, batch_size):
	packed_instances = []
	packed_instance = []
	packed_lengths = []
	packed_length = []
	packed_idx = 0
	max_length = len(trn_instances[0][0])
	for instance in trn_instances:
		if packed_idx != 0 and packed_idx % batch_size == 0:
			packed_instances.append(packed_instance)
			packed_lengths.append(packed_length)
			packed_instance = []
			packed_length = []
		if len(packed_instance) == 0:
			max_length = len(instance[0])

		packed_length.append(len(instance[0]))
		instance[0] += [[0] for i in range(max_length-len(instance[0]))]
		instance[1] += [[0] for i in range(max_length-len(instance[1]))]
		instance[2] += [[0] for i in range(max_length-len(instance[2]))]
		instance[3] += [0 for i in range(max_length-len(instance[3]))]
		packed_instance.append(instance)
		
		packed_idx += 1
	if len(packed_instance) != 0:
		packed_instances.append(packed_instance)
		packed_lengths.append(packed_length)
	return packed_instances, packed_lengths

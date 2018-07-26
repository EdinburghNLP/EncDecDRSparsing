#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script computes dmatch F-score between two DRGs. It is based on and very similar to SMATCH.
For detailed description of smatch, see http://www.isi.edu/natural-language/drg/smatch-13.pdf

As opposed to AMRs, this script takes the triples directly as input. Each triple should be on a
single line, with DRGs separated by a newline. Lines starting with '%' or '#" are ignored. 
Sample input looks like this:


% DRG 1

b1 REF x1
b2 REF e1
b1 person.n.01 c1
c1 ARG1 x1
b1 Named c2 % Pierce 0-6
c2 ARG1 x1
c2 ARG2 "pierce"
b2 populate.v.01 c3 % lives 7-12

% DRG 2

c5 ARG1 x2
b3 Named c6 % Rossville~Blvd 18-32
c6 ARG1 x2
c6 ARG2 "rossville~boulevard"
b2 Topic c7 % near 13-17

Command line options:

-f1   : First file with DRG triples, usually produced file
-f2   : Second file with DRG triples, usually gold file
-r	  : Number of restarts used
-m    : Max number of triples for a DRG to still take them into account - default 0 means no limit
-p    : Number of parallel threads to use (default 1)
-runs : Number of runs to average over, if you want a more reliable result
-s    : What kind of smart initial mapping we use:
      -no No smart mappings
      -order Smart mapping based on order of the variables in the DRG (b1 maps to b1, b2 to b2, etc)
      -conc  Smart mapping based on matching concepts (their match is likely to be in the optimal mapping)
      -att   Smart mapping based on attribute triples (proper names usually), default
      -init  Smart mapping based on number of initial matches for a set of mappings
      -freq  Smart mapping based on frequency of the variables (currently not implemented)
-prin : Print more specific output, such as individual (average) F-scores for the smart initial mappings, and the matching and non-matching triples
-sense: Use this to do sense experiments
      -normal Don't change anything, just use triples as is (default)   
      -wrong  Always use the wrong sense for a concept - used to see impact of concept identification
      -ignore Ignore sense - meaning we always produced the correct sense
      -base   Always use the first sense for a concept (baseline)
-sig  : Number of significant digits to output (default 4)
-b    : Use this for baseline experiments, comparing a single DRG to a list of DRGs. Produced DRG file should contain a single DRG.
-ms   : Instead of averaging the score, output a score for each DRG
-pr   : Also output precison and recall
-v    : Verbose output
-vv   : Very verbose output      
"""

import os, random, time, argparse, re, sys, codecs
reload(sys)
sys.setdefaultencoding('utf-8')	#necessary to avoid unicode errors
import multiprocessing
from multiprocessing import Pool
import psutil #for memory usage


def get_memory_usage():
     #return the memory usage in MB
	process = psutil.Process(os.getpid())
	mem = process.memory_info()[0] / float(1000000)
	return mem


def get_best_match(attribute1, relation1, attribute2, relation2,
				   prefix1, prefix2, var_count, var_types1, var_types2, var_map_prod, var_map_gold, prin, single, significant, memory_limit, counter):
	"""
	Get the highest triple match number between two sets of triples via hill-climbing.
	Arguments:
		attribute1: attribute triples of DRG 1 (attribute name, node name, attribute value)
		relation1: relation triples of DRG 1 (relation name, node 1 name, node 2 name)
		attribute2: attribute triples of DRG 2 (attribute name, node name, attribute value)
		relation2: relation triples of DRG 2 (relation name, node 1 name, node 2 name)
		prefix1: prefix label for DRG 1
		prefix2: prefix label for DRG 2
	Returns:
		best_match: the node mapping that results in the highest triple matching number
		best_match_num: the highest triple matching number

	"""
	# Compute candidate pool - all possible node match candidates.
	# In the hill-climbing, we only consider candidate in this pool to save computing time.
	# weight_dict is a dictionary that maps a pair of node
	(candidate_mappings, weight_dict) = compute_pool(attribute1, relation1, attribute2, relation2,
													 prefix1, prefix2, var_count, var_types1, var_types2, var_map_prod, var_map_gold)
	
	test_triple_num =  len(attribute1) + len(relation1)	
	gold_triple_num =  len(attribute2) + len(relation2)
	match_triple_dict = {}	#save mapping and number of matches so that we don't have to calculate stuff twice
	
	if veryVerbose:
		print >> DEBUG_LOG, "Candidate mappings:"
		print >> DEBUG_LOG, candidate_mappings
		print >> DEBUG_LOG, "Weight dictionary"
		print >> DEBUG_LOG, weight_dict
	
	# initialize best match mapping
	# the ith entry is the node index in DRG 2 which maps to the ith node in DRG 1
	best_match_num  = 0
	best_mapping 	= [-1] * len(var_map_prod)
	total_triples 	=  len(attribute1) + len(relation1)
	found_idx 		= 0
	random_mappings = []
	all_fscores 	= []
	
	smart_mappings 	= get_smart_mappings(candidate_mappings, attribute1, attribute2, relation1, relation2, weight_dict, args.restarts, args.smart, match_triple_dict)	#get smart mappings
	num_smart		= len(smart_mappings)
	smart_fscores 	= [0] * num_smart
	
	mapping_list 	= get_mapping_list(candidate_mappings, weight_dict, args.restarts - num_smart, False, match_triple_dict)		#get random mappings, or best mapping + random mappings in order depending on args.best_map
	mapping_order 	= smart_mappings + mapping_list
	
	for i, map_cur in enumerate(mapping_order):   #number of restarts is number of mappings
		cur_mapping = map_cur[0]
		match_num = map_cur[1]
		#print 'initial length', len(cur_mapping)
		if veryVerbose:
			print >> DEBUG_LOG, "Node mapping at start", cur_mapping
			print >> DEBUG_LOG, "Triple match number at start:", match_num
		
		while True:
			# get best gain
			(gain, new_mapping) = get_best_gain(cur_mapping, candidate_mappings, weight_dict, len(var_map_gold), match_num, match_triple_dict)
			if veryVerbose:
				print >> DEBUG_LOG, "Gain after the hill-climbing", gain
			# hill-climbing until there will be no gain for new node mapping
			if gain <= 0:
				break
			# otherwise update match_num and mapping
			match_num += gain
			cur_mapping = new_mapping[:]
			if veryVerbose:
				print >> DEBUG_LOG, "Update triple match number to:", match_num
				print >> DEBUG_LOG, "Current mapping:", cur_mapping
			
		if match_num > best_match_num:
			best_mapping = cur_mapping[:]
			best_match_num = match_num
			found_idx = i
		
		if get_memory_usage() > memory_limit:		#check if we exceed maximum memory, if so, clear triple_dict
			match_triple_dict.clear() 				#saves us from getting memory issues with high number of restarts
			
		if best_match_num == total_triples and i > num_smart - 1:		#if we have matched as much we can (precision 1.0), we might as well stop instead of doing all other restarts - but always do smart mappings
			(precision, recall, f_score) = compute_f(best_match_num, test_triple_num, gold_triple_num, significant, False)
			all_fscores.append(f_score)
			if prin and single:
				print 'Best match already found, stop restarts at restart {0}'.format(i)
			if i < len(smart_fscores):
				smart_fscores[i] = best_match_num
			break
		
		(precision, recall, f_score) = compute_f(match_num, test_triple_num, gold_triple_num, significant, False)
		all_fscores.append(f_score)
		
		if i < len(smart_fscores):		#are we still adding smart F-scores?
			smart_fscores[i] = best_match_num
	
	avg_f = round(float(sum(all_fscores)) / float(len(all_fscores)), args.significant)	#average F-score over all restarts
	match_triple_dict.clear() #clear matches out of memory
			
	return best_mapping, best_match_num, found_idx, len(mapping_order), avg_f, smart_fscores


def get_smart_mappings(candidate_mappings, attribute1, attribute2,relation1, relation2, weight_dict, restarts, smart, match_triple_dict):
	'''Get all the smart mappings specified
	no   : don't do any smart mapping
	order: match variables based on the order they occur in
	init : Get 1000 random mappings, take the one with the most initial matches
	conc : Smart mapping based on matching concepts
	freq : Smart mapping based on frequency of variables
	all  : All previously described mappings'''
	
	if smart == 'no':
		return []
	elif smart == 'att':	#get mapping based on attribute matches		
		smart_attr = smart_att_mapping(candidate_mappings, attribute1, attribute2)
		smart_mappings_all = [[smart_attr, compute_match(smart_attr, weight_dict, match_triple_dict)]]
	elif smart == 'order':		#get mapping based on variable order
		smart_order = smart_order_mapping(candidate_mappings)
		smart_mappings_all = [[smart_order, compute_match(smart_order, weight_dict, match_triple_dict)]]
	elif smart == 'init':	#get mapping that initially had the most matches (out of set of 1000)
		smart_init = get_mapping_list(candidate_mappings, weight_dict, restarts, True, match_triple_dict)
		smart_mappings_all = [[smart_init, compute_match(smart_init, weight_dict, match_triple_dict)]]
	elif smart == 'conc':
		smart_conc = smart_concept_mapping(candidate_mappings, relation1, relation2)
		smart_mappings_all = [[smart_conc, compute_match(smart_conc, weight_dict, match_triple_dict)]]
	elif smart == 'freq':
		raise NotImplementedError("Sorry, frequency mapping not implemented yet")
	elif smart == 'all':
		smart_attr = smart_att_mapping(candidate_mappings, attribute1, attribute2)
		smart_order = smart_order_mapping(candidate_mappings)
		smart_init = get_mapping_list(candidate_mappings, weight_dict, restarts, True, match_triple_dict)
		smart_conc = smart_concept_mapping(candidate_mappings, relation1, relation2)
		smart_mappings_all = [[smart_attr, compute_match(smart_attr, weight_dict, match_triple_dict)],[smart_order, compute_match(smart_order, weight_dict, match_triple_dict)], [smart_init, compute_match(smart_init, weight_dict, match_triple_dict)], [smart_conc, compute_match(smart_conc, weight_dict, match_triple_dict)]]
	else:	#argparse should take care of this
		return []
	
	return smart_mappings_all


def get_mapping_list(candidate_mappings, weight_dict, total_restarts, select_best, match_triple_dict):
	'''Function that returns a mapping list, each item being [mapping, match_num]. If best_first is true, it returns
	   a mapping with first a percentage of mappings that initially already had a lot of matches.'''
	
	if total_restarts <= 0:		#nothing to do here, we do more smart mappings than restarts anyway
		return []
	
	random_maps = []
	map_ceil = 1000 if select_best else total_restarts	#ceiling is 1000 if we try to find the best initial mapping, if random it is just the total we want
	
	random_maps = get_random_set(candidate_mappings, weight_dict, map_ceil, match_triple_dict)	#get set of random mappings here
	
	if select_best:	#we only want the best random mapping	
		sorted_maps = sorted(random_maps, key=lambda x: x[1], reverse = True)	#sort mappings by number of initial matches
		if sorted_maps:
			return sorted_maps[0][0]	#return best mapping
		else:
			print 'No best init mapping found'
			return []	
	else:
		return random_maps


def get_random_set(candidate_mappings, weight_dict, map_ceil, match_triple_dict):
	'''Function that returns a set of random mappings based on candidate_mappings'''
	
	random_maps = []
	count_duplicate = 0
	
	while len(random_maps) < map_ceil and count_duplicate <= 50:	#only do random mappings we haven't done before, but if we have 100 duplicates in a row, we are probably done with all mappings
		cur_mapping = random_init_mapping(candidate_mappings)
		match_num = compute_match(cur_mapping, weight_dict, match_triple_dict)
		if [cur_mapping, match_num] not in random_maps:
			random_maps.append([cur_mapping, match_num])
			count_duplicate = 0
		else:	
			count_duplicate += 1

	return random_maps
	
	
def normalize(item):
	"""
	lowercase and remove quote signifiers from items that are about to be compared
	"""
	item = item.lower().rstrip('_')
	return item


def compute_pool(attribute1, relation1, attribute2, relation2,
				 prefix1, prefix2, var_count, var_types1, var_types2, var_map_prod, var_map_gold):
	"""
	compute all possible node mapping candidates and their weights (the triple matching number gain resulting from
	mapping one node in DRG 1 to another node in AMR2)

	Arguments:
		attribute1: attribute triples of DRG 1 (attribute name, node name, attribute value)
		relation1: relation triples of DRG 1 (relation name, node 1 name, node 2 name)
		attribute2: attribute triples of DRG 2 (attribute name, node name, attribute value)
		relation2: relation triples of DRG 2 (relation name, node 1 name, node 2 name
		var_count: number of variables
		var_types1: var-types of DRG1
		var_types1: var-types of DRG2
		prefix1: prefix label for DRG 1
		prefix2: prefix label for DRG 2
	Returns:
	  candidate_mapping: a list of candidate nodes.
					   The ith element contains the node indices (in DRG 2) the ith node (in DRG 1) can map to.
					   (resulting in non-zero triple match)
	  weight_dict: a dictionary which contains the matching triple number for every pair of node mapping. The key
				   is a node pair. The value is another dictionary. key {-1} is triple match resulting from this node
				   pair alone (attribute triples), and other keys are node pairs that can result
				   in relation triple match together with the first node pair.


	"""
	candidate_mapping = []
	weight_dict = {}

	for i in range(var_count):			#add as much candidate mappings as we have variables
		candidate_mapping.append(set())
	
	for i in range(0, len(attribute1)):
		for j in range(0, len(attribute2)):
			# if both attribute relation triple have the same relation name and value 
			if normalize(attribute1[i][0]) == normalize(attribute2[j][0]) \
			   and normalize(attribute1[i][2]) == normalize(attribute2[j][2]):
				if var_types1[attribute1[i][1]] == var_types2[attribute2[j][1]]:	#does the variable type match? -- should always match I think
					node1_index = int(attribute1[i][1][len(prefix1):])
					node2_index = int(attribute2[j][1][len(prefix2):])
					candidate_mapping[node1_index].add(node2_index)
					node_pair = (node1_index, node2_index)
					# use -1 as key in weight_dict for attribute triples
					if node_pair in weight_dict:
						weight_dict[node_pair][-1] += 1
					else:
						weight_dict[node_pair] = {}
						weight_dict[node_pair][-1] = 1	
	
	for i in range(0, len(relation1)): 
		for j in range(0, len(relation2)):
			# if both relation share the same name
			if normalize(relation1[i][0]) == normalize(relation2[j][0]):
				if var_types1[relation1[i][1]] == var_types2[relation2[j][1]] and var_types1[relation1[i][2]] == var_types2[relation2[j][2]]: 	#check if nodes are even allowed to match, based on their type
					
					node1_index_drg1 = int(relation1[i][1][len(prefix1):])
					node1_index_drg2 = int(relation2[j][1][len(prefix2):])
					node2_index_drg1 = int(relation1[i][2][len(prefix1):])
					node2_index_drg2 = int(relation2[j][2][len(prefix2):])
					
					# add mapping between two nodes
					candidate_mapping[node1_index_drg1].add(node1_index_drg2)
					candidate_mapping[node2_index_drg1].add(node2_index_drg2)
					node_pair1 = (node1_index_drg1, node1_index_drg2)
					node_pair2 = (node2_index_drg1, node2_index_drg2)
					if node_pair2 != node_pair1:
						# update weight_dict weight. Note that we need to update both entries for future search
						# i.e weight_dict[node_pair1][node_pair2]
						#     weight_dict[node_pair2][node_pair1]
						#if node1_index_drg1 > node2_index_drg1:  #Rik removed this because he thinks it is not necessary to swap if you add both anyway
							# swap node_pair1 and node_pair2
						#	node_pair1 = (node2_index_drg1, node2_index_drg2)
						#	node_pair2 = (node1_index_drg1, node1_index_drg2)
						
						if node_pair1 in weight_dict:
							if node_pair2 in weight_dict[node_pair1]:
								weight_dict[node_pair1][node_pair2] += 1
							else:
								weight_dict[node_pair1][node_pair2] = 1
						else:
							weight_dict[node_pair1] = {}
							weight_dict[node_pair1][-1] = 0
							weight_dict[node_pair1][node_pair2] = 1
						if node_pair2 in weight_dict:
							if node_pair1 in weight_dict[node_pair2]:
								weight_dict[node_pair2][node_pair1] += 1
							else:
								weight_dict[node_pair2][node_pair1] = 1
						else:
							weight_dict[node_pair2] = {}
							weight_dict[node_pair2][-1] = 0
							weight_dict[node_pair2][node_pair1] = 1
					else:
						# two node pairs are the same. So we only update weight_dict once.
						# this generally should not happen.
						if node_pair1 in weight_dict:
							weight_dict[node_pair1][-1] += 1
						else:
							weight_dict[node_pair1] = {}
							weight_dict[node_pair1][-1] = 1
	
	return candidate_mapping, weight_dict


def smart_att_mapping(candidate_mapping, attribute1, attribute2):
	"""
	Initialize mapping based on the concept mapping (smart initialization)
	Arguments:
		candidate_mapping: candidate node match list
		attribute1: attribute triples of DRG 1
		attribute2: attribute triples of DRG 2
	Returns:
		initialized node mapping between two AMRs

	"""
	random.seed()
	matched_dict = {}
	result = []
	# list to store node indices that have no concept match
	no_word_match = []
	smart_match = 0
	
	for i, candidates in enumerate(candidate_mapping):
		if len(candidates) == 0:
			# no possible mapping
			result.append(-1)
			continue
		# node value in attribute triples of DRG 1
		for att in attribute1:
			if int(att[1][1:]) == i:
				#found attribute triple for this variable
				value1 = att[2]
				arg1 = att[0]
				
				for node_index in candidates:
					for att2 in attribute2:
						if int(att2[1][1:]) == node_index:	#attribute exists for this node index
							value2 = att2[2]
							arg2 = att2[0]
						# find the first attribute triple match in the candidates
						# attribute triple match is having the same concept value
							if value1 == value2 and arg1 == arg2:	#both should match for a smart mapping match
								if node_index not in matched_dict:
									smart_match += 1
									result.append(node_index)
									matched_dict[node_index] = 1
									break
		if len(result) == i:
			no_word_match.append(i)
			result.append(-1)
	
	# if no concept match, generate a random mapping
	for i in no_word_match:
		candidates = list(candidate_mapping[i])
		while len(candidates) > 0:
			# get a random node index from candidates
			rid = random.randint(0, len(candidates) - 1)
			if candidates[rid] in matched_dict:
				candidates.pop(rid)
			else:
				matched_dict[candidates[rid]] = 1
				result[i] = candidates[rid]
				break	
	
	return result


def smart_concept_mapping(candidate_mapping, relation1, relation2):
	"""
	Initialize mapping based on the concept mapping (smart initialization)
	Arguments:
		candidate_mapping: candidate node match list
		relation1: relation triples of DRG 1
		relation2: relation triples of DRG 2
	Returns:
		initialized node mapping between two AMRs

	"""
	random.seed()
	matched_dict = {}
	result = []
	# list to store node indices that have no concept match
	no_word_match = []
	smart_match = 0
	
	for i, candidates in enumerate(candidate_mapping):
		if len(candidates) == 0:
			# no possible mapping
			result.append(-1)
			continue
		# node value in relation triples of DRG 1
		for rel in relation1:
			if int(rel[1][1:]) == i and len(re.findall(r'\.[a-z]\.[\d]+', rel[0])) > 0:
				#found relation triple for this variable + matches concept (e.g. WORD.a.00)
				concept1, var1, var2 = rel
				matched = False #not matched yet
				for node_index in candidates:
					for rel2 in relation2:
						if int(rel2[1][1:]) == node_index and not matched:	#relation exists for this node index
							concept2, var3, var4 = rel2
							# find the first relation triple match in the candidates
							# relation triple match is having the same concept value
							if concept1 == concept2:	#both should match for a smart mapping match
								if int(var4.replace('b','')) in candidate_mapping[int(rel[2][1:])]:	#second variable also needs to match (possibly)
									if node_index not in matched_dict:	#only add if we did not match this node yet
										smart_match += 1
										result.append(node_index)
										matched_dict[node_index] = 1
										matched = True
		if len(result) == i:
			no_word_match.append(i)	#failed to match for this variable
			result.append(-1)
	
	# if no concept match, generate a random mapping
	for i in no_word_match:
		candidates = list(candidate_mapping[i])
		while len(candidates) > 0:
			# get a random node index from candidates
			rid = random.randint(0, len(candidates) - 1)
			if candidates[rid] in matched_dict:
				candidates.pop(rid)
			else:
				matched_dict[candidates[rid]] = 1
				result[i] = candidates[rid]
				break	
	
	return result

	
def smart_order_mapping(candidate_mapping):
	"""
	Initialize mapping based on order of the variables (smart initialization)
	Arguments:
		candidate_mapping: candidate node match list
	Returns:
		initialized node mapping between two AMRs

	"""
	matched_dict = {}
	result = []
	# list to store node indices that have no concept match
	no_word_match = [i for i in range(len(candidate_mapping))]
	result = [-1 for i in range(len(candidate_mapping))]
	
	
	for i in no_word_match:
		candidates = list(candidate_mapping[i])
		poss_candidates = []
		for c in candidates:				#only check for possible candidates
			if c not in matched_dict:
				poss_candidates.append(c)
		
		if poss_candidates:
			for idx, pc in enumerate(poss_candidates):
				diff = abs(pc - i)			#calculate difference in variable order, lower is better
				if idx == 0:
					best_diff = diff
					best_map = pc
				elif diff < best_diff:
					best_diff = diff
					best_map = pc
			
			matched_dict[best_map] = 1		#update mapping 
			result[i] = best_map
	
	return result
		

def random_init_mapping(candidate_mapping):
	"""
	Generate a random node mapping.
	Args:
		candidate_mapping: candidate_mapping: candidate node match list
	Returns:
		randomly-generated node mapping between two AMRs

	"""
	# if needed, a fixed seed could be passed here to generate same random (to help debugging)
	random.seed()
	matched_dict = {}
	result = []
	for c in candidate_mapping:
		candidates = list(c)
		if len(candidates) == 0:
			# -1 indicates no possible mapping
			result.append(-1)
			continue
		found = False
		while len(candidates) > 0:
			# randomly generate an index in [0, length of candidates)
			rid = random.randint(0, len(candidates) - 1)
			# check if it has already been matched
			if candidates[rid] in matched_dict:
				candidates.pop(rid)
			else:
				matched_dict[candidates[rid]] = 1
				result.append(candidates[rid])
				found = True
				break
		if not found:
			result.append(-1)
	return result

 
def compute_match(mapping, weight_dict, match_triple_dict):
	"""
	Given a node mapping, compute match number based on weight_dict.
	Args:
	mappings: a list of node index in DRG 2. The ith element (value j) means node i in DRG 1 maps to node j in DRG 2.
	Returns:
	matching triple number
	Complexity: O(m*n) , m is the node number of DRG 1, n is the node number of DRG 2

	"""
	# If this mapping has been investigated before, retrieve the value instead of re-computing.
	if veryVerbose:
		print >> DEBUG_LOG, "Computing match for mapping"
		print >> DEBUG_LOG, mapping
	if tuple(mapping) in match_triple_dict:
		#print 'Break search because we saw this before'
		if veryVerbose:
			print >> DEBUG_LOG, "saved value", match_triple_dict[tuple(mapping)]
		return match_triple_dict[tuple(mapping)]
	match_num = 0
	# i is node index in DRG 1, m is node index in DRG 2
	for i, m in enumerate(mapping):
		if m == -1:
			# no node maps to this node
			continue
		# node i in DRG 1 maps to node m in DRG 2
		current_node_pair = (i, m)
		if current_node_pair not in weight_dict:
			continue
		if veryVerbose:
			print >> DEBUG_LOG, "node_pair", current_node_pair
		for key in weight_dict[current_node_pair]:
			if key == -1:
				# matching triple resulting from attribute triples
				match_num += weight_dict[current_node_pair][key]
				if veryVerbose:
					print >> DEBUG_LOG, "attribute match", weight_dict[current_node_pair][key]
			# only consider node index larger than i to avoid duplicates
			# as we store both weight_dict[node_pair1][node_pair2] and
			#     weight_dict[node_pair2][node_pair1] for a relation
			elif key[0] < i:
				continue
			elif mapping[key[0]] == key[1]: #key is also a node pair. If we have e.g. node_pair(1,5), we check if indeed mapping[1] is 5, meaning that in the current mapping 1 maps to 5
				match_num += weight_dict[current_node_pair][key]
				if veryVerbose:
					print >> DEBUG_LOG, "relation match with", key, weight_dict[current_node_pair][key]
	if veryVerbose:
		print >> DEBUG_LOG, "match computing complete, result:", match_num
	# update match_triple_dict
	match_triple_dict[tuple(mapping)] = match_num
	return match_num  


def move_gain(mapping, node_id, old_id, new_id, weight_dict, match_num, match_triple_dict):
	"""
	Compute the triple match number gain from the move operation
	Arguments:
		mapping: current node mapping
		node_id: remapped node in DRG 1
		old_id: original node id in DRG 2 to which node_id is mapped
		new_id: new node in to which node_id is mapped
		weight_dict: weight dictionary
		match_num: the original triple matching number
	Returns:
		the triple match gain number (might be negative)

	"""
	# new node mapping after moving
	new_mapping = (node_id, new_id)
	# node mapping before moving
	old_mapping = (node_id, old_id)
	# new nodes mapping list (all node pairs)
	new_mapping_list = mapping[:]
	new_mapping_list[node_id] = new_id
	# if this mapping is already been investigated, use saved one to avoid duplicate computing
	if tuple(new_mapping_list) in match_triple_dict:
		return match_triple_dict[tuple(new_mapping_list)] - match_num
	gain = 0
	# add the triple match incurred by new_mapping to gain
	if new_mapping in weight_dict:
		for key in weight_dict[new_mapping]:
			if key == -1:
				# attribute triple match
				gain += weight_dict[new_mapping][-1]
			elif new_mapping_list[key[0]] == key[1]:
				# relation gain incurred by new_mapping and another node pair in new_mapping_list
				gain += weight_dict[new_mapping][key]
	# deduct the triple match incurred by old_mapping from gain
	if old_mapping in weight_dict:
		for k in weight_dict[old_mapping]:
			if k == -1:
				gain -= weight_dict[old_mapping][-1]
			elif mapping[k[0]] == k[1]:
				gain -= weight_dict[old_mapping][k]
	# update match number dictionary
	match_triple_dict[tuple(new_mapping_list)] = match_num + gain
	return gain


def swap_gain(mapping, node_id1, mapping_id1, node_id2, mapping_id2, weight_dict, match_num, match_triple_dict):
	"""
	Compute the triple match number gain from the swapping
	Arguments:
	mapping: current node mapping list
	node_id1: node 1 index in DRG 1
	mapping_id1: the node index in DRG 2 node 1 maps to (in the current mapping)
	node_id2: node 2 index in DRG 1
	mapping_id2: the node index in DRG 2 node 2 maps to (in the current mapping)
	weight_dict: weight dictionary
	match_num: the original matching triple number
	Returns:
	the gain number (might be negative)

	"""
	new_mapping_list = mapping[:]
	# Before swapping, node_id1 maps to mapping_id1, and node_id2 maps to mapping_id2
	# After swapping, node_id1 maps to mapping_id2 and node_id2 maps to mapping_id1
	new_mapping_list[node_id1] = mapping_id2
	new_mapping_list[node_id2] = mapping_id1
	if tuple(new_mapping_list) in match_triple_dict:
		return match_triple_dict[tuple(new_mapping_list)] - match_num
	gain = 0
	new_mapping1 = (node_id1, mapping_id2)
	new_mapping2 = (node_id2, mapping_id1)
	old_mapping1 = (node_id1, mapping_id1)
	old_mapping2 = (node_id2, mapping_id2)
	if node_id1 > node_id2:
		new_mapping2 = (node_id1, mapping_id2)
		new_mapping1 = (node_id2, mapping_id1)
		old_mapping1 = (node_id2, mapping_id2)
		old_mapping2 = (node_id1, mapping_id1)
	if new_mapping1 in weight_dict:
		for key in weight_dict[new_mapping1]:
			if key == -1:
				gain += weight_dict[new_mapping1][-1]
			elif new_mapping_list[key[0]] == key[1]:
				gain += weight_dict[new_mapping1][key]
	if new_mapping2 in weight_dict:
		for key in weight_dict[new_mapping2]:
			if key == -1:
				gain += weight_dict[new_mapping2][-1]
			# to avoid duplicate
			elif key[0] == node_id1:
				continue
			elif new_mapping_list[key[0]] == key[1]:
				gain += weight_dict[new_mapping2][key]
	if old_mapping1 in weight_dict:
		for key in weight_dict[old_mapping1]:
			if key == -1:
				gain -= weight_dict[old_mapping1][-1]
			elif mapping[key[0]] == key[1]:
				gain -= weight_dict[old_mapping1][key]
	if old_mapping2 in weight_dict:
		for key in weight_dict[old_mapping2]:
			if key == -1:
				gain -= weight_dict[old_mapping2][-1]
			# to avoid duplicate
			elif key[0] == node_id1:
				continue
			elif mapping[key[0]] == key[1]:
				gain -= weight_dict[old_mapping2][key]
	match_triple_dict[tuple(new_mapping_list)] = match_num + gain
	
	return gain


def get_best_gain(mapping, candidate_mappings, weight_dict, num_vars, cur_match_num, match_triple_dict):
	"""
	Hill-climbing method to return the best gain swap/move can get
	Arguments:
	mapping: current node mapping
	candidate_mappings: the candidates mapping list
	weight_dict: the weight dictionary
	num_vars: the number of the nodes in DRG 2
	cur_match_num: current triple match number
	Returns:
	the best gain we can get via swap/move operation

	"""
	largest_gain = 0
	# True: using swap; False: using move
	use_swap = True
	# the node to be moved/swapped
	node1 = None
	# store the other node affected. In swap, this other node is the node swapping with node1. In move, this other
	# node is the node node1 will move to.
	node2 = None
	# unmatched nodes in DRG 2
	unmatched = set(range(0, num_vars))
	# exclude nodes in current mapping
	# get unmatched nodes
	for nid in mapping:
		if nid in unmatched:
			unmatched.remove(nid)
	
	for i, nid in enumerate(mapping):
		# current node i in DRG 1 maps to node nid in DRG 2
		for nm in unmatched:
			if nm in candidate_mappings[i]:
				# remap i to another unmatched node (move)
				# (i, m) -> (i, nm)
				if veryVerbose:
					print >> DEBUG_LOG, "Remap node", i, "from ", nid, "to", nm
				mv_gain = move_gain(mapping, i, nid, nm, weight_dict, cur_match_num, match_triple_dict)
				if veryVerbose:
					print >> DEBUG_LOG, "Move gain:", mv_gain
					new_mapping = mapping[:]
					new_mapping[i] = nm
					new_match_num = compute_match(new_mapping, weight_dict, match_triple_dict)
					if new_match_num != cur_match_num + mv_gain:
						print >> ERROR_LOG, mapping, new_mapping
						print >> ERROR_LOG, "Inconsistency in computing: move gain", cur_match_num, mv_gain, \
							new_match_num
				if mv_gain > largest_gain:
					largest_gain = mv_gain
					node1 = i
					node2 = nm
					use_swap = False
	# compute swap gain

	for i, m in enumerate(mapping):
		for j in range(i+1, len(mapping)):
			m2 = mapping[j]
			# swap operation (i, m) (j, m2) -> (i, m2) (j, m)
			# j starts from i+1, to avoid duplicate swap
			
			if veryVerbose:
				print >> DEBUG_LOG, "Swap node", i, "and", j
				print >> DEBUG_LOG, "Before swapping:", i, "-", m, ",", j, "-", m2
				print >> DEBUG_LOG, mapping
				print >> DEBUG_LOG, "After swapping:", i, "-", m2, ",", j, "-", m
			sw_gain = swap_gain(mapping, i, m, j, m2, weight_dict, cur_match_num, match_triple_dict)
			if veryVerbose:
				print >> DEBUG_LOG, "Swap gain:", sw_gain
				new_mapping = mapping[:]
				new_mapping[i] = m2
				new_mapping[j] = m
				print >> DEBUG_LOG, new_mapping
				new_match_num = compute_match(new_mapping, weight_dict, match_triple_dict)
				if new_match_num != cur_match_num + sw_gain:
					print >> ERROR_LOG, match, new_match
					print >> ERROR_LOG, "Inconsistency in computing: swap gain", cur_match_num, sw_gain, new_match_num
			if sw_gain > largest_gain:
				largest_gain = sw_gain
				node1 = i
				node2 = j
				use_swap = True		
	# generate a new mapping based on swap/move
	cur_mapping = mapping[:]
	if node1 is not None:
		if use_swap:
			if veryVerbose:
				print >> DEBUG_LOG, "Use swap gain"
			temp = cur_mapping[node1]
			cur_mapping[node1] = cur_mapping[node2]
			cur_mapping[node2] = temp
		else:
			if veryVerbose:
				print >> DEBUG_LOG, "Use move gain"
			cur_mapping[node1] = node2
	else:
		if veryVerbose:
			print >> DEBUG_LOG, "no move/swap gain found"

	if veryVerbose:
		print >> DEBUG_LOG, "Original mapping", mapping
		print >> DEBUG_LOG, "Current mapping", cur_mapping
		print 'Largest gain', largest_gain
	
	return largest_gain, cur_mapping


def compute_f(match_num, test_num, gold_num, significant, f_only):
	"""
	Compute the f-score based on the matching triple number,
								 triple number of DRG set 1,
								 triple number of DRG set 2
	Args:
		match_num: matching triple number
		test_num:  triple number of DRG 1 (test file)
		gold_num:  triple number of DRG 2 (gold file)
	Returns:
		precision: match_num/test_num
		recall: match_num/gold_num
		f_score: 2*precision*recall/(precision+recall)
	"""
	if test_num == 0 or gold_num == 0:
		if f_only:
			return 0.00
		else:
			return 0.00, 0.00, 0.00
	precision = round(float(match_num) / float(test_num), significant)
	recall = round(float(match_num) / float(gold_num), significant)
	if (precision + recall) != 0:
		f_score = round(2 * precision * recall / (precision + recall), significant)
		if veryVerbose:
			print >> DEBUG_LOG, "F-score:", f_score
		if f_only:
			return f_score
		else:
			return precision, recall, f_score
	else:
		if veryVerbose:
			print >> DEBUG_LOG, "F-score:", "0.0"
		if f_only:
			return 0.00
		else:
			return precision, recall, 0.00


def fix_equ_triples(triple_list):
	'''If we have the triples (b EQU c), (c ARG1 x),  (c ARG2 y), this is equivalent to
	   (b EQU c), (c ARG1 y),  (c ARG2 x). Solution: change to (b EQU c), (c ARG x),  (c ARG y) 
	   so that order is not important'''
	
	for idx1, triples in enumerate(triple_list):
		equ_triples = [x[2] for x in triples if x[1] == 'EQU'] #save variable that has EQU triple
		
		for equ in equ_triples:			#check if we have to replace ARG1/ARG2 in some triples
			for idx2, t in enumerate(triples):
				if t[0] == equ  and (t[1] == 'ARG1' or t[1] == 'ARG2'):  #found equ variable again, now with ARG1 or ARG2 edge
					triple_list[idx1][idx2] = (t[0], 'ARG', t[2])		 #change triple here, ARG1/ARG2 to ARG

	return triple_list											


def get_triples(f, sense, prod_bool):
	'''Function that returns a list of DRGs (that consists of triples)'''
	
	if prod_bool:	#print sense information to screen if we don't do normal
		if sense == 'wrong' and prod_bool:
			print 'Doing sense experiment in which sense of concepts is always wrong (.00)\n'
		elif sense == 'base' and prod_bool:
			print 'Doing sense experiment in which sense of concepts is always baseline (.01)\n'	
		elif sense == 'ignore':
			print 'Doing sense experiment in which we ignore sense of concepts (so always correct)\n'
	
	triple_list = []
	cur_triples = []
	
	with open(f,'r') as in_f:
		input_lines = in_f.read().split('\n')
		for idx, line in enumerate(input_lines):
			if line.strip().startswith('%') or line.strip().startswith('#'):
				pass #skip comments
			elif not line.strip():
				if cur_triples:	#newline, so DRG is finished, add to list. Ignore double/triple newlines	
					triple_list.append(cur_triples)
				cur_triples = []
			else:
				try:
					# For sense = 'wrong', we only do something for the produced file - change sense to 0 (always incorrect).
					# For sense = 'ignore', we ignore the sense, so we just remove the sense information
					
					triple = [line.split()[0], line.split()[1], line.split()[2]]
						
					if sense == 'wrong' and prod_bool:
						if triple[1].count('.') == 2:
							triple[1] = ".".join(triple[1].split('.')[0:-1]) + '.00'	
					elif sense == 'base' and prod_bool:
						if triple[1].count('.') == 2:
							triple[1] = ".".join(triple[1].split('.')[0:-1]) + '.01'		
					elif sense == 'ignore':
						if triple[1].count('.') == 2:
							triple[1] = triple[1].split('.')[0]
						
					cur_triples.append(triple)			
				except:
					print 'Triple incorrectly format at line {0}\nFor line: {1}\nExiting...'.format(idx, line)
					sys.exit(1)	
	
	if cur_triples:		#no newline at the end, still add the DRG
		triple_list.append(cur_triples)
	
	## Change triples if they have an EQU edge occurs in the list ##
	
	triple_list = fix_equ_triples(triple_list)
	
	return triple_list			



def between_quotes(t):
	'''Return true if third triple value is between quotes'''
	return (t.startswith('"') and t.endswith('"')) or ( t.startswith("'") and t.endswith("'"))


def get_var_type(t):
	'''Function that returns the type of a variable, B, C, P or X'''
	if t.startswith('b') or t.startswith('k'):
		var_type = 'B'
	elif t.startswith('c'):
		var_type = 'C'
	elif t.startswith('p'):
		var_type = 'P'
	else:
		var_type = 'X'
	
	return var_type	


def rename_var(t, vars_seen, type_vars, prefix, var_map):
	'''Function that renames the variables in a standardized way'''
	
	if t in vars_seen:
		new_var = prefix + str(vars_seen.index(t))
	else:
		new_var = prefix + str(len(vars_seen))
		var_type = get_var_type(t)
		type_vars[new_var] = var_type
		vars_seen.append(t)	
	
	var_map[new_var] = t
	
	return new_var, vars_seen, type_vars, var_map	

def get_specific_triples(triple_list, prefix):
	'''Function that gets the specific attribute and relation triples
	   Also renames them and changes their format to DRG format'''
	
	attr 		= []
	rel 		= []
	vars_seen 	= []
	type_vars 	= {}
	var_map 	= {}
	
	for triple in triple_list:
		if between_quotes(triple[2]):			#attribute triple
			val0, vars_seen, type_vars, var_map = rename_var(triple[0], vars_seen, type_vars, prefix, var_map)
			new_triple = (triple[1], val0, triple[2])		#reformat to DRG style triples
			if new_triple not in attr:
				attr.append(new_triple)
		else:							#relation triple	
			val0, vars_seen, type_vars, var_map = rename_var(triple[0], vars_seen, type_vars, prefix, var_map)
			val2, vars_seen, type_vars, var_map = rename_var(triple[2], vars_seen, type_vars, prefix, var_map)
			new_triple = triple[1], val0, val2			 	#reformat to DRG style triples
			if new_triple not in rel:
				rel.append(new_triple)	
	
	return attr, rel, len(vars_seen), type_vars, var_map		


def fill_baseline_list(baseline_drg, triples_gold_list):
	'''Fill baseline drgs so that we can all DRGs to a baseline'''
	new_prod = []
	
	for x in range(len(triples_gold_list)):
		new_prod.append(baseline_drg)
	
	return new_prod


def rewritten_triples(triples, new_mapping, count):
	''''Rewrite set of triples to their new shared representation based on their best mapping'''
	
	new_triples = []
	
	for t in triples:
		arg, var1, var2 = t
		
		if var1 in new_mapping:
			var1 = new_mapping[var1]
		elif (var1.startswith('a') or var1.startswith('b')) and var1[-1].isdigit():	#missed variable, still add
			new_mapping[var1] = 'c' + str(count)
			count += 1
		
		if var2 in new_mapping:
			var2 = new_mapping[var2]	
		elif (var2.startswith('a') or var2.startswith('b')) and var2[-1].isdigit():	#missed variable, still add
			new_mapping[var1] = 'c' + str(count)
			count += 1
		new_triples.append((arg, var1, var2))
		
	return new_triples, new_mapping, count


def get_dict_val(key, d):
	'''If item occurs in a dictionary we return the dict value, else just the item'''
	if key in d:
		return d[key]
	else:
		return key	


def check_triple_match(new_prod_triples, new_gold_triples, attributes1, relation1, attributes2, relation2, var_map_prod, var_map_gold):
	'''Check which specific triples matched in our best mapping'''
	
	match 		 = []
	no_match 	 = []
	matched_gold = []
	
	new_gold_triples_lower = [(x.lower(), y.lower(), z.lower()) for (x,y,z) in new_gold_triples]	#comparison has to be in lower case
	
	## find matched triples ##
	
	for idx, p in enumerate(new_prod_triples):
		prod_arg, prod_var1, prod_var2 = (attributes1 + relation1)[idx]
		
		##  map variables to their original variable ##
		
		prod_var1 = get_dict_val(prod_var1, var_map_prod)
		prod_var2 = get_dict_val(prod_var2, var_map_prod)
		compare_triple = (p[0].lower(), p[1].lower(), p[2].lower())	#comparison has to be in lower case
		found = False
		
		## add matching variables here ##
		for idx2, gold_triple in enumerate(new_gold_triples_lower):
			gold_arg, gold_var1, gold_var2 = (attributes2 + relation2)[idx2]
			
			gold_var1 = get_dict_val(gold_var1, var_map_gold)
			gold_var2 = get_dict_val(gold_var2, var_map_gold)
			
			if compare_triple == gold_triple:
				match.append([" ".join([str(x) for x in (prod_var1, prod_arg, prod_var2)]), " ".join([str(x) for x in (gold_var1, gold_arg, gold_var2)])])	#add matched triple for both languages
				matched_gold.append(new_gold_triples[idx2])
				found = True
				break
		
		if not found:	#no match found, added to be compared tripled to non-matched list
			no_match.append([" ".join([str(x) for x in (prod_var1, prod_arg, prod_var2)]), ''])
	
	## find non-matched triples for gold file ##
	
	added = 0

	for idx, gold_triple in enumerate(new_gold_triples):	#add gold non-matched printed to no_match so that they are also printed
		gold_arg, gold_var1, gold_var2 = (attributes2 + relation2)[idx]	
		gold_var1 = get_dict_val(gold_var1, var_map_gold)
		gold_var2 = get_dict_val(gold_var2, var_map_gold)
		
		if gold_triple not in matched_gold:					#was never matched
			if added < len(no_match):
				no_match[added][1] = " ".join([str(x) for x in (gold_var1, gold_arg, gold_var2)])	#switch order back to original order
				added += 1
			else:
				no_match.append([''," ".join([str(x) for x in (gold_var1, gold_arg, gold_var2)])])
	
	return match, no_match		


def match_drg(drg_file, drg_print):
	'''Match new DRG with the original DRG so that we can retrieve the comments again (not always useful, but Boxer does this for example)'''
	for orig in drg_file:
		if orig.startswith(drg_print):
			return orig.strip()
	
	## HACKY: if we are here, maybe we replaced ARG1/ARG2 by ARG at some point due to the EQU triples. We still want to the return the original then
	
	for orig in drg_file:
		if orig.replace('ARG1','ARG').replace('ARG2','ARG').startswith(drg_print):
			return orig.strip()
	
	## Otherwise we really can't find the original, something must be wrong
	
	raise ValueError("Triple in new set that could not be found in original set")
	


def restore_original(drg_lines, prod_drg, gold_drg):
	'''Restore original DRG lines with comments for nicer overview on site'''
	new_lines = []
	
	for line in drg_lines:
		part = ['','']
		for idx, drg_print in enumerate(line):
			if idx == 0 and drg_print:
				part[idx] = match_drg(prod_drg, drg_print)
			elif idx == 1 and drg_print:
				part[idx] = match_drg(gold_drg, drg_print)
		new_lines.append(part)
		del part
	
	assert len(new_lines) == len(drg_lines), "When restoring original we end up with wrong length of new DRG"
	
	return new_lines


def create_tab_list(print_rows, print_item):
	'''For multiple rows, return row of strings nicely separated by tabs'''
	
	col_widths = []
	return_rows = [print_item]
	if print_rows:
		for idx in range(len(print_rows[0])):					#for nice printing, calculate max column length
			col_widths.append(max([len(x[idx]) for x in print_rows]) + 1)
		
		for idx, row in enumerate(print_rows):					#print rows here, adjusted for column width
			return_rows.append("| ".join(word.ljust(col_widths[col_idx]) for col_idx, word in enumerate(row)))
		
		for r in return_rows:
			print r
		
	return return_rows
		


def print_mapping(best_mapping, attributes1, relation1, attributes2, relation2,  var_map_prod, var_map_gold, precision, recall, prin, prod_file, gold_file):
	'''Print the matching and non-matching triples in our final mapping - either to screen or to a file'''
	
	new_mapping = {}
	
	for var1, var2 in enumerate(best_mapping):			#create new shared mapping: both a1 and b3 match to c3, for example. Don't add mapping for -1, means no mapping found
		new_mapping['a' + str(var1)] = 'c' + str(var1)
		if var2 != -1:
			new_mapping['b' + str(var2)] = 'c' + str(var1)
			#print var_map_prod['a' + str(var1)], 'maps to', var_map_gold['b' + str(var2)] #uncomment to print mapping of original variables
	
	if not best_mapping:
		var1 = 0
	
	new_prod_triples, new_mapping, count = rewritten_triples(attributes1 + relation1, new_mapping, var1 + 1)
	new_gold_triples, new_mapping, count = rewritten_triples(attributes2 + relation2, new_mapping, count + 1)
	
	match, no_match = check_triple_match(new_prod_triples, new_gold_triples, attributes1, relation1, attributes2, relation2,  var_map_prod, var_map_gold)
	
	prec = round(float(len(match)) / float(len(new_prod_triples)),4)
	rec =  round(float(len(match)) / float(len(new_gold_triples)),4)
	
	if prec != precision or rec != recall:
		raise ValueError("Error in calculating precision and recall when doing triple match - different values in outcome")

	if prin:	#print specific info to screen
		
		## get original drgs ##
		prod_drg   = [x.strip() for x in open(prod_file,'r')]
		gold_drg   = [x.strip() for x in open(gold_file,'r')]
		
		## restore original comments or other information after triple ##
		new_match  	 = restore_original(match, prod_drg, gold_drg)
		new_no_match = restore_original(no_match, prod_drg, gold_drg)
		
		#print them nicely separated by tabs
		print_match    = create_tab_list(new_match, '\nMatching triples:\n')
		print_no_match = create_tab_list(new_no_match, '\nNon-matching triples:\n')
	
	return prec, rec	
			
			
def get_matching_triples(arg_list):
	'''Function that gets matching triples (easier to parallelize)'''
	start_time = time.time()
	prod_t, gold_t, args, prefix1, prefix2, significant, prin, prod_file, gold_file, single, ms, smart, restarts, max_triples, memory_limit, counter = arg_list #unpack argument list
	
	attributes1, relation1, var_count1, var_types1, var_map_prod = get_specific_triples(prod_t, prefix1)		# prefixes 'a' and 'b' are used to create standardized variable-names
	attributes2, relation2, var_count2, var_types2, var_map_gold = get_specific_triples(gold_t, prefix2)
	num_triples = len(attributes1) + len(relation1)
	
	if single and (args.max_triples > 0 and (len(attributes1) + len(relation1) > args.max_triples or len(attributes2) + len(relation2) > args.max_triples)):
		print 'Skip calculation of DRG, triples longer than max of {0}'.format(args.max_triples)
		return []
	
	if verbose:
		# print parse results of two AMRs
		print >> DEBUG_LOG, "DRS pair"
		print >> DEBUG_LOG, "============================================"
		print >> DEBUG_LOG, "Attribute triples of DRG 1:", len(attributes1)
		print >> DEBUG_LOG, attributes1
		print >> DEBUG_LOG, "Relation triples of DRG 1:", len(relation1)
		print >> DEBUG_LOG, relation1
		print >> DEBUG_LOG, "Attribute triples of DRG 2:", len(attributes2)
		print >> DEBUG_LOG, attributes2
		print >> DEBUG_LOG, "Relation triples of DRG 2:", len(relation2)
		print >> DEBUG_LOG, relation2
	(best_mapping, best_match_num, found_idx, restarts_done, avg_f, smart_fscores) = get_best_match(attributes1, relation1, attributes2, relation2,
													prefix1, prefix2, var_count1, var_types1, var_types2, var_map_prod, var_map_gold, prin, single, significant, memory_limit, counter)
																								
	if verbose:
		print >> DEBUG_LOG, "best match number", best_match_num
		print >> DEBUG_LOG, "best node mapping", best_mapping
	
	test_triple_num =  len(attributes1) + len(relation1)	
	gold_triple_num =  len(attributes2) + len(relation2)
	
	(precision, recall, best_f_score) = compute_f(best_match_num, test_triple_num, gold_triple_num, significant, False)
	if prin and (args.ms or single) and not args.no_mapping: 
		#print triple mapping if -prin is used and we either do multiple scores, or just had a single DRG	
		print_mapping(best_mapping, attributes1, relation1, attributes2, relation2, var_map_prod, var_map_gold, precision, recall, prin, prod_file, gold_file)
	if args.ms and not single: #for single DRG we print results later on anyway
		print_results([[best_match_num, test_triple_num, gold_triple_num, smart_fscores, avg_f, restarts_done, found_idx]], False, start_time, prin, ms, smart, restarts, significant, max_triples, single)
	return [best_match_num, test_triple_num, gold_triple_num, smart_fscores, avg_f, restarts_done, found_idx]


def print_results(res_list, no_print, start_time, prin, ms, smart, restarts, significant, max_triples, single):
	'''Print the final or inbetween scores -- res_list has format [[best_match_num, test_triple_num, total_gold_num, smart_fscores, avg_f, restarts_done, found_idx]]'''
	
	extra_print = '' if ms else 'Average' #add Average in printing when doing a single score
	
	total_match_num = sum([x[0] for x in res_list if x])
	total_test_num  = sum([x[1] for x in res_list if x])
	total_gold_num  = sum([x[2] for x in res_list if x])
	avg_f 			= round(float(sum([x[4] for x in res_list if x])) / float(len([x[4] for x in res_list if x])), significant) 
	restarts_done	= round(float(sum([x[5] for x in res_list if x])) / float(len([x[5] for x in res_list if x])), significant) 
	found_idx		= round(float(sum([x[6] for x in res_list if x])) / float(len([x[6] for x in res_list if x])), significant) 
	runtime 		= round(time.time() - start_time, significant)
	
	if verbose:
		print >> DEBUG_LOG, "Total match number, total triple number in DRS 1, and total triple number in DRS 2:"
		print >> DEBUG_LOG, total_match_num, total_test_num, total_gold_num
		print >> DEBUG_LOG, "---------------------------------------------------------------------------------"
	
	## output document-level dmatch score (a single f-score for all DRG pairs in two files) ##
	
	(precision, recall, best_f_score) = compute_f(total_match_num, total_test_num, total_gold_num, significant, False)
	
	if not res_list:
		return []  #no results for some reason
	elif no_print: #averaging over multiple runs, don't print results
		return [precision, recall, best_f_score]
	else:
		print '\n## Triple information ##\n'
		print 'Triples prod : {0}'.format(total_test_num)	
		print 'Triples gold : {0}\n'.format(total_gold_num)
		print 'Max number of triples per DRG (0 means no limit): {0}\n'.format(max_triples)
		print '## Main Results ##\n'
		if not single:
			print 'All shown number are averages calculated over {0} DRG-pairs\n'.format(len(res_list))
		print 'Matching triples: {0}\n'.format(total_match_num)
		if pr_flag:
			print "Precision: {0}".format(round(precision, significant))
			print "Recall   : {0}".format(round(recall, significant))
			print "F-score  : {0}".format(round(best_f_score, significant))
		else:
			print "F-score  : {0}".format(round(best_f_score, significant))
		
		if prin:	#print specific output here
			print '\n## Detailed F-scores ##\n'
			if smart != 'no':
				smart_first = compute_f(sum([y[0] for y in [x[3] for x in res_list]]), total_test_num, total_gold_num, significant, True)
			
			if smart == 'all':
				smart_att   = compute_f(sum([y[0] for y in [x[3] for x in res_list]]), total_test_num, total_gold_num, significant, True)
				smart_order = compute_f(sum([y[1] for y in [x[3] for x in res_list]]), total_test_num, total_gold_num, significant, True)
				smart_init  = compute_f(sum([y[2] for y in [x[3] for x in res_list]]), total_test_num, total_gold_num, significant, True)
				smart_conc  = compute_f(sum([y[3] for y in [x[3] for x in res_list]]), total_test_num, total_gold_num, significant, True)
				
				print 'Smart attribute F-score      : {0}'.format(smart_att)
				print 'Smart order F-score          : {0}'.format(smart_order)
				print 'Smart F-score initial matches: {0}'.format(smart_init)
				print 'Smart F-score concepts       : {0}'.format(smart_conc)
			elif smart == 'att':
				print 'Smart attribute F-score: {0}'.format(smart_first)
			elif smart == 'order':		
				print 'Smart order F-score: {0}'.format(smart_first)
			elif smart == 'init':
				print 'Smart F-score initial matches: {0}'.format(smart_first)
			elif smart == 'conc':
				print 'Smart F-score concepts: {0}'.format(smart_first)
			
			if single:	#averaging does not make for multiple DRGs	
				print 'Avg F-score over all restarts: {0}'.format(avg_f)
			
			print '\n## Restarts and processing time ##\n'
			
			print 'Num restarts specified       : {0}'.format(restarts)
			print 'Num restarts performed       : {0}'.format(restarts_done)
			print 'Found best mapping at restart: {0}'.format(int(found_idx))
			print 'Total processing time        : {0} sec'.format(runtime)

	return [precision, recall, best_f_score]

			
def main(arguments):
	"""
	Main function of dmatch score calculation

	"""
	global verbose
	global veryVerbose
	global single_score
	global pr_flag
	
	start = time.time()
	
	if arguments.ms:
		single_score = False
	if arguments.v:
		verbose = True
	if arguments.vv:
		veryVerbose = True
	if arguments.pr:
		pr_flag = True
	
	total_match_num, total_test_num, total_gold_num = 0, 0, 0	#triple number that matches, #triple number in test,  #triple number in gold	
	sent_num = 1
	prefix1 = 'a'
	prefix2 = 'b'
	
	## get the triples ##
	
	triples_prod_list = get_triples(args.f1, args.sense, True)
	triples_gold_list = get_triples(args.f2, args.sense, False)
	
	no_print = True if args.runs > 1 else False 			#don't print the results each time if we do multiple runs
	single = True if len(triples_gold_list) == 1 else False #we are doing a single DRG
	
	## check if correct input ##
	
	if args.baseline:		#if we try a baseline DRG, we have to fill a list of this baseline
		if len(triples_prod_list) == 1:
			print 'Testing baseline DRGs vs {0} DRGs...\n'.format(len(triples_gold_list))
			triples_prod_list = fill_baseline_list(triples_prod_list[0], triples_gold_list)
		else:
			raise ValueError("Using --baseline, but there is more than 1 DRG in prod file")
	elif len(triples_prod_list) != len(triples_gold_list):
		print "Number of DRGs not equal, {0} vs {1}, exiting...".format(len(triples_prod_list), len(triples_gold_list))
		sys.exit(0)
	elif len(triples_prod_list) == 0 and len(triples_gold_list) == 0:
		print "Both DRGs empty, exiting..."
		sys.exit(0)
	elif not single:
		print '\nComparing {0} DRGs...\n'.format(len(triples_gold_list))	
	
	counter = 0
	res = []
	
	if args.max_triples > 0 and not single:	#print number of DRGs we skip due to the -max_triples parameter
		print 'Skipping {0} DRGs due to their length exceeding {1} (-max_triples)\n'.format(len([x for x,y in zip(triples_gold_list, triples_prod_list) if (len(x) > args.max_triples or len(y) > args.max_triples)]), args.max_triples)
	
	## Processing triples ##
	
	for idx in range(args.runs):	#for experiments we want to more runs so we can average later
		
		all_results = []	#save as [best_match_num, test_triple_num, total_gold_num, smart_fscores]
		arg_list = []		#list with all calls
		counter = 0
		for prod_t, gold_t in zip(triples_prod_list, triples_gold_list):		
			counter += 1
			arg_list.append([prod_t, gold_t, args, prefix1, prefix2, args.significant, args.prin, args.f1, args.f2, single, args.ms, args.smart, args.restarts, args.max_triples, args.mem_limit, counter])
		
		## parallel processing here
		if args.parallel == 1: #no need for parallel processing for 1 thread
			all_results = []
			len_lines = 0
			for c, a in enumerate(arg_list):
				len_lines += (len(a[0]) + 4)
				all_results.append(get_matching_triples(a))
		else:
			all_results = multiprocessing.Pool(args.parallel).map(get_matching_triples, arg_list)
		
		if all_results:
			if all_results[0]:
				res.append(print_results(all_results, no_print, start, args.prin, args.ms, args.smart, args.restarts, args.significant, args.max_triples, single))
			else:
				print 'No results found, exiting..'	
		else:
			print 'No results found, exiting..'		
	
	## If multiple runs, print averages ##
	
	if res and args.runs > 1:
		print 'Average scores over {0} runs:\n'.format(args.runs)
		print 'Precision: {0}'.format(round(float(sum([x[0] for x in res])) / float(args.runs), args.significant))
		print 'Recall   : {0}'.format(round(float(sum([x[1] for x in res])) / float(args.runs), args.significant))
		print 'F-score  : {0}'.format(round(float(sum([x[2] for x in res])) / float(args.runs), args.significant))


def build_arg_parser():

	parser = argparse.ArgumentParser(description="D-match calculator -- arguments")
	
	parser.add_argument('-f1', required=True, type=str,
						help='First file with DRG triples, DRGs need to be separated by blank line')
	parser.add_argument('-f2', required=True, type=str,
						help='Second file with DRG triples, DRGs need to be separated by blank line')
	parser.add_argument('-m','--max_triples', type=int, default=0, help='Maximum number of triples for DRG (default 0 means no limit)')
	parser.add_argument('-p', '--parallel', type=int, default=1, help='Number of parallel threads we use (default 1)')
	parser.add_argument('-runs', type=int, default=1, help='Usually we do 1 run, only for experiments we can increase the number of runs to get a better average')
	parser.add_argument('-s', '--smart', default = 'all', action='store', choices=['no','all','order','freq','conc', 'att', 'init'], help='What kind of smart mapping do we use (default all)')
	parser.add_argument('-prin', action='store_true', help='Print very specific output - matching and non-matching triples and specific F-scores for smart mappings')
	parser.add_argument('-sense', default = 'normal', action='store', choices=['normal','wrong', 'ignore','base',], help='How do we evaluate sense - wrong means sense is always wrong, for "ignore" sense is always correct, "base" means always adding -01 for prod file. Default is normal.')					
	parser.add_argument('-r','--restarts', type=int, default=100, help='Restart number (default: 100)')
	parser.add_argument('-sig', '--significant', type=int, default=4, help='significant digits to output (default: 4)')
	parser.add_argument('-mem', '--mem_limit', type=int, default=500, help='Memory limit in MBs (default 500). Note that this is per parallel thread! If you use -par 4, each thread gets 500 MB with default settings.')
	parser.add_argument('-b', '--baseline', action='store_true', default=False, help="Helps in deciding a good baseline DRG. If added, prod-file must be a single DRG, gold file a number of DRGs to compare to (default false)")
	parser.add_argument('-v', action='store_true', help='Verbose output (Default:false)')
	parser.add_argument('-vv', action='store_true', help='Very Verbose output (Default:false)')
	parser.add_argument('-ms', action='store_true', default=False,
						help='Output multiple scores (one pair per score)' \
							 'instead of a single document-level dmatch score (Default: false)')
	parser.add_argument('-pr', action='store_true', default=False,
						help="Output precision and recall as well as the f-score. Default: false")
	parser.add_argument('-nm','--no_mapping', action='store_true', help='Do not print the mapping information')
	args = parser.parse_args()
	
	## 	Check if files exist ##
	
	if not os.path.exists(args.f1):
		raise ValueError("File for -f1 does not exist")
	if not os.path.exists(args.f2):
		raise ValueError("File for -f2 does not exist")
	
	## Check if combination of arguments is valid ##
	
	if args.ms and args.runs > 1:
		raise NotImplementedError("Not implemented to average over individual scores, only use -ms when doing a single run")
	
	if args.restarts < 1:
		raise ValueError('Number of restarts must be larger than 0')
	elif args.smart == 'all' and args.restarts < 4:
		raise ValueError("All smart mappings is already 4 restarts, therefore -r should at least be 4, not {0}".format(args.restarts))	
	
	if args.ms and args.parallel > 1:
		print 'WARNING: using -ms and -par > 1 messes up printing to screen - not recommended'
		time.sleep(5) #so people can still read the warning
	
	elif (args.vv or args.v) and args.parallel > 1:
		print 'WARNING: using -vv or -v and par > 1 messes up printing to screen - not recommended'
		time.sleep(5) #so people can still read the warning	
	
	if args.runs > 1 and args.prin:
		print 'WARNING: we do not print specific information (-prin) for runs > 1, only final averages'
		time.sleep(5)
	
	return args	

# verbose output switch.
# Default false (no verbose output)
verbose = False
veryVerbose = False

# single score output switch.
# Default true (compute a single score for all AMRs in two files)
single_score = True

# precision and recall output switch.
# Default false (do not output precision and recall, just output F score)
pr_flag = False

# Error log location
ERROR_LOG = sys.stderr

# Debug log location
DEBUG_LOG = sys.stderr

if __name__ == "__main__":
	args = build_arg_parser()
	main(args)


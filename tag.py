
import re
###global_relation
class Tag:
	def __init__(self, filename, lemmas):
		self.filename = filename

		#14 8 39 14 13
		self.MAX_KV = 15
		self.MAX_PV = 10
		self.MAX_XV = 40
		self.MAX_EV = 15
		self.MAX_SV = 15

		self.SOS = "<SOS>"
		self.EOS = "<EOS>"
		self.CARD = "CARD_NUMBER"
		self.TIME = "TIME_NUMBER"
		self.reduce = ")"
		
		self.act_rel_k = "GEN_REL_K"
		self.act_rel_p = "GEN_REL_P"
		self.act_tag_k = "GEN_TAG_K"
		self.act_tag_p = "GEN_TAG_P"
		self.act_tag_x = "GEN_TAG_X"
		self.act_tag_e = "GEN_TAG_E"
		self.act_tag_s = "GEN_TAG_S"

		self.act_rel_global = "REL_GLOBAL"
		self.act_rel_local = "REL_LOCAL"

		self.rel_sdrs = "SDRS("
		self.rel_drs = "DRS("
		self.rel_not = "NOT("
		self.rel_nec = "NEC("
		self.rel_pos = "POS("
		self.rel_or = "OR("
		self.rel_duplex = "DUPLEX("
		self.rel_imp = "IMP("
		self.rel_timex = "TIMEX("
		self.rel_card = "CARD("
		self.rel_eq = "EQ("

		self.relation_global = list()
		for line in open(filename):
			line = line.strip()
			if line[0] == "#":
				continue
			self.relation_global.append(line.strip().upper())
		
		self.tag_to_ix = {self.SOS:0, self.EOS:1, self.CARD:2, self.TIME:3}
		self.ix_to_tag = [self.SOS, self.EOS, self.CARD, self.TIME]
		
		
		self.tag_to_ix[self.reduce] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.reduce) #4
		self.tag_to_ix[self.rel_sdrs] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_sdrs) #5
		self.tag_to_ix[self.rel_drs] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_drs) #6
		self.tag_to_ix[self.rel_not] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_not) #7
		self.tag_to_ix[self.rel_nec] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_nec) #8
		self.tag_to_ix[self.rel_pos] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_pos) #9
		self.tag_to_ix[self.rel_or] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_or) #10
		self.tag_to_ix[self.rel_duplex] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_duplex) #11
		self.tag_to_ix[self.rel_imp] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_imp) #12

		self.tag_to_ix[self.rel_timex] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_timex) #13
		self.tag_to_ix[self.rel_card] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_card) #14
		self.tag_to_ix[self.rel_eq] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_eq) #15

		self.global_start = len(self.tag_to_ix)
		for tag in self.relation_global:
			if tag in self.tag_to_ix:
				continue
			self.tag_to_ix[tag] = len(self.tag_to_ix)
			self.ix_to_tag.append(tag)

		#self.tag_to_ix[self.act_rel_k] = len(self.tag_to_ix)
		#self.ix_to_tag.append(self.act_rel_k)
		self.k_rel_start = len(self.tag_to_ix)
		for i in range(self.MAX_KV):
			self.tag_to_ix["K"+str(i+1)+"("] = len(self.tag_to_ix)
			self.ix_to_tag.append("K"+str(i+1)+"(")
		self.p_rel_start = len(self.tag_to_ix)
		for i in range(self.MAX_PV):
			self.tag_to_ix["P"+str(i+1)+"("] = len(self.tag_to_ix)
			self.ix_to_tag.append("P"+str(i+1)+"(")
		self.k_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_KV):
			self.tag_to_ix["K"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("K"+str(i+1))
		self.p_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_PV):
			self.tag_to_ix["P"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("P"+str(i+1))
		self.x_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_XV):
			self.tag_to_ix["X"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("X"+str(i+1))
		self.e_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_EV):
			self.tag_to_ix["E"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("E"+str(i+1))
		self.s_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_SV):
			self.tag_to_ix["S"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("S"+str(i+1))

		self.tag_size = len(self.tag_to_ix)
		assert len(self.tag_to_ix) == len(self.ix_to_tag)

		self.UNK = "<UNK>("
		self.ix_to_lemma = list()
		for lemma in lemmas:
			assert lemma not in self.tag_to_ix
			self.tag_to_ix[lemma+"("] = len(self.tag_to_ix)
			self.ix_to_lemma.append(lemma+"(")
		self.all_tag_size = len(self.tag_to_ix)

	def type(self, string):
		if string in self.ix_to_tag:
			return -2, self.tag_to_ix[string]
		else:
			return -1, -1
"""
	def get_var_represent(self, targets, lemmas):
		re = [ [i+self.x_tag_start] for i in range(self.tag_size - self.x_tag_start)]
		i = 0
		while i < len(targets):
			x = targets[i]
			relation = False
			if x[0] != -2:
				lemma = lemmas[x[0]]
				relation = True
			else:
				lemma = x[1]
				if x[1] >= 13 and x[1] < self.k_rel_start:
					relation = True

			if relation:
				assert i+1 < len(tokens) and target[i+1][0] == -2 and target[i+1][1] < self.tag_size and target[i+1][1] >= self.x_tag_start:
				re[targets[i+1][1] - self.x_tag_start].append(lemma)

				if i + 2 < len(targets) and target[i+2][0] == -2 and target[i+2][1] < self.tag_size and target[i+2][1] >= self.x_tag_start:
					re[targets[i+2][1] - self.x_tag_start].append(lemma)

		return re
"""
					
			
				
				
				





		




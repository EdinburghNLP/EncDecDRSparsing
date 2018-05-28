
class OuterMask:
	def __init__(self, tags_info):
		self.tags_info = tags_info
		self.mask = 0
		self.need = 1

		self.SOS = tags_info.tag_size*10
		self.relation = tags_info.tag_size*10+1

		self.variable_offset = 0
		self.relation_offset = 1
		self.k_relation_offset = 2
		self.p_relation_offset = 3
		self.drs_offset = 4
		self.six_offset = 5
		
		self.reset()
	def reset(self, ):
		self.relation_count = 0
		self.stack = [self.SOS]
		self.stack_ex = [[0 for i in range(6)]]

		self.stack_variables = []
		self.k = 1
		self.p = 1
		self.x = 1
		self.e = 1
		self.s = 1

	def get_all_mask(self, inputs):
		res = []
		#self._print_state()
		res.append(self.get_step_mask())
		#print res[-1]
		for type, ix in inputs:
			assert res[-1][ix] != self.mask
			self.update(type, ix)
			#self._print_state()
			res.append(self.get_step_mask())
			#print res[-1]
		return res

	def get_step_mask(self):
		if self.stack[-1] == self.SOS:
			#SOS
			return self._get_sos_mask()
		elif self.stack[-1] == 5:
			#SDRS
			return self._get_sdrs_mask()
		elif self.stack[-1] == 6:
			#DRS
			return self._get_drs_mask()
		elif self.stack[-1] in [7, 8, 9]:
			#not, nec, pos
			return self._get_1_mask()
		elif self.stack[-1] in [10, 11, 12]:
			#or, imp, duplex
			return self._get_2_mask()
		elif self.stack[-1] == self.tags_info.k_rel_start or self.stack[-1] == self.tags_info.p_rel_start:
			#k p
			return self._get_1_mask()
		else:
			assert False
	def _get_sos_mask(self):
		if self.stack_ex[-1][self.drs_offset] == 0:
			re = self._get_zeros(self.tags_info.tag_size)
			re[self.tags_info.tag_to_ix[self.tags_info.rel_drs]] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[1] = self.need
			return re
	def _get_sdrs_mask(self):
		#SDRS
		if self.stack_ex[-1][self.k_relation_offset] < 2:
			#only k
			re = self._get_zeros(self.tags_info.tag_size)
			idx = self.tags_info.k_rel_start + self.k - 1
			re[idx] = self.need
			return re
		else:
			#only reduce
			re = self._get_zeros(self.tags_info.tag_size)
			re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
			if self.relation_count <= 40:
				cnt = 0
				for i in range(len(self.stack)-1):
					if self.stack[i] == 5 and self.stack_ex[i][self.k_relation_offset] == 0:
						cnt += 1
				if self.k <= self.tags_info.MAX_KV - cnt:
					# k is ok
					idx = self.tags_info.k_rel_start + self.k - 1
					re[idx] = self.need
			return re
	def _get_drs_mask(self):
		re = self._get_zeros(self.tags_info.tag_size)
		re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
		if self.relation_count <= 40:
			if self.p <= self.tags_info.MAX_PV:
				idx = self.tags_info.p_rel_start + self.p - 1
				re[idx] = self.need
			re[7] = self.need
			re[8] = self.need
			re[9] = self.need
			re[10] = self.need
			re[11] = self.need
			re[12] = self.need
		return re
	def _get_1_mask(self):
		if self.stack_ex[-1][self.drs_offset] == 0:
			re = self._get_zeros(self.tags_info.tag_size)
			if self.relation_count <= 40:
				cnt = 0
				for i in range(len(self.stack)):
					if self.stack[i] == 5 and self.stack_ex[i][self.k_relation_offset] == 0:
						cnt += 1
				if self.k + 1 <= self.tags_info.MAX_KV - cnt: #enough k to produce sdrs
					re[5] = self.need
			re[6] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[4] = self.need
			return re
	def _get_2_mask(self):
		if self.stack_ex[-1][self.drs_offset] <= 1:
			re = self._get_zeros(self.tags_info.tag_size)
			if self.relation_count <= 40:
				cnt = 0
				for i in range(len(self.stack)):
					if self.stack[i] == 5 and self.stack_ex[i][self.k_relation_offset] == 0:
						cnt += 1
				if self.k + 1 <= self.tags_info.MAX_KV - cnt: # enough k to produce sdrs
					re[5] = self.need
			re[6] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[4] = self.need
			return re

	def update(self, type, ix):
		if ix >= 5 and ix <= 12:
			self.stack.append(ix)
			self.relation_count += 1
			self.stack_ex.append([0 for i in range(6)])
		elif ix >= self.tags_info.k_rel_start and ix < self.tags_info.p_rel_start:
			self.stack.append(self.tags_info.k_rel_start)
			self.relation_count += 1
			self.stack_ex.append([0 for i in range(6)])
			self.k += 1
		elif ix >= self.tags_info.p_rel_start and ix < self.tags_info.k_tag_start:
			self.stack.append(self.tags_info.p_rel_start)
			self.relation_count += 1
			self.stack_ex.append([0 for i in range(6)])
			self.p += 1
		elif ix == 4:
			self.stack_ex.pop()
			if self.stack[-1] == 5 or self.stack[-1] == 6:
				self.stack_ex[-1][self.drs_offset] += 1
			elif self.stack[-1] >= 7 and self.stack[-1] <= 12:
				self.stack_ex[-1][self.six_offset] += 1
			elif self.stack[-1] == self.tags_info.k_rel_start:
				self.stack_ex[-1][self.k_relation_offset] += 1
			elif self.stack[-1] == self.tags_info.p_rel_start:
				self.stack_ex[-1][self.p_relation_offset] += 1
			else:
				assert False
			self.stack.pop()
		elif ix == 1:
			pass
		else:
			assert False

	def _print_state(self):
		print "relation_count", self.relation_count
		print "stack", self.stack
		print "stack_ex", self.stack_ex
		print "stack_variables", self.stack_variables
		print "kpxes", self.k, self.p, self.x, self.e, self.s
	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]

class RelationMask:
	#sdrs should have at least two k(), at least one relation, and the relation should follow k()
	#drs should have at least anything, except variables.
	#not, nec, pos should have and only have one drs or sdrs
	#imp, or, duplex should have and only have two drs or sdrs
	#timex should be timex(variables, TIME_NUMBER)
	#card should be card(variables, CARD_NUMBER)
	#k(, p( should have and only have one drs or sdrs
	#variable constrains
	def __init__(self, tags_info, encoder_input_size=0):
		self.tags_info = tags_info
		self.mask = 0
		self.need = 1
		
		self.reset(encoder_input_size)
	def reset(self, encoder_input_size):
		self.encoder_input_size = encoder_input_size
		self.is_sdrs = False

	def set_sdrs(self, is_sdrs):
		self.is_sdrs = is_sdrs

	def _get_relations(self):
		res = self._get_zeros(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
		res[1] = self.need
		idx = 13
		if self.is_sdrs:
			idx = self.tags_info.global_start
		while idx < self.tags_info.k_rel_start:
			res[idx] = self.need
			idx += 1
		return res

	def get_all_mask(self, input_size, least):
		res = []
		for i in range(input_size+1):
			res.append(self._get_relations())
		if least:
			res[0][1] = self.mask # next of condition
		return res
	def get_step_mask(self, least):
		relations = self._get_relations()
		if least:
			relations[1] = self.mask
		return relations
	
	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]

class VariableMask:
	#sdrs should have at least two k(), at least one relation, and the relation should follow k()
	#drs should have at least anything, except variables.
	#not, nec, pos should have and only have one drs or sdrs
	#imp, or, duplex should have and only have two drs or sdrs
	#timex should be timex(variables, TIME_NUMBER)
	#card should be card(variables, CARD_NUMBER)
	#k(, p( should have and only have one drs or sdrs
	#variable constrains
	def __init__(self, tags_info):
		self.tags_info = tags_info
		self.mask = 0
		self.need = 1
		self.k_use = False

		self.reset(0)
	def reset(self, p_max, k_use=False):
		#self.k = 1
		self.p_max = p_max
		self.x = 1
		self.e = 1
		self.s = 1

		self.stack_ex = []
		self.stack = []
		self.stack_drs = []

		self.prev_variable = -1
		self.pre_prev_variable = -1

		self.user_k = []
		self.k_use = k_use
	def set_k(self, user_k):
		self.user_k = user_k

	def get_step_mask(self):
		if self.stack_drs[-1] == 5:
			return self._get_sdrs_mask()
		else:
			return self._get_drs_mask()

	def _get_sdrs_mask(self):
		#SDRS
		if self.prev_variable == -1:
			re = self._get_zeros(self.tags_info.tag_size)
			if self.k_use:
				for idx in self.user_k:
					re[idx + self.tags_info.k_tag_start] = self.need
			else:
				for idx in self.stack_ex[-1]:
					re[idx] = self.need

			return re
		elif self.prev_prev_variable == -1:
			re = self._get_zeros(self.tags_info.tag_size)
			if self.k_use:
				for idx in self.user_k:
					re[idx + self.tags_info.k_tag_start] = self.need
			else:
				for idx in self.stack_ex[-1]:
					re[idx] = self.need
			assert self.prev_variable != -1
			re[self.prev_variable] = self.mask
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[1] = self.need
			return re

	def _get_drs_mask(self):
		if self.prev_variable == -1:
			re = self._get_zeros(self.tags_info.tag_size)

			idx = self.tags_info.x_tag_start
			while idx < self.tags_info.e_tag_start and idx < self.tags_info.x_tag_start + self.x:
				re[idx] = self.need
				idx += 1
			idx = self.tags_info.e_tag_start
			while idx < self.tags_info.s_tag_start and idx < self.tags_info.e_tag_start + self.e:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.s_tag_start
			while idx < self.tags_info.tag_size and idx < self.tags_info.s_tag_start + self.s:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.p_tag_start
			while idx < self.tags_info.x_tag_start and idx < self.tags_info.p_tag_start + self.p_max:
				re[idx] = self.need
				idx += 1
			#while idx < self.tags_info.p_tag_start + self.max_p and idx < self.tags_info.p_tag_start + self.p:
			#	re[idx] = self.need
			#	idx += 1

			return re
		elif self.prev_prev_variable == -1:
			re = self._get_zeros(self.tags_info.tag_size)
			re[1] = self.need

			if self.stack[-1] == 13:
				re[3] = self.need
			elif self.stack[-1] == 14:
				re[2] = self.need
			else:
				idx = self.tags_info.x_tag_start
				while idx < self.tags_info.e_tag_start and idx < self.tags_info.x_tag_start + self.x:
					re[idx] = self.need
					idx += 1
				idx = self.tags_info.e_tag_start
				while idx < self.tags_info.s_tag_start and idx < self.tags_info.e_tag_start + self.e:
					re[idx] = self.need
					idx += 1

				idx = self.tags_info.s_tag_start
				while idx < self.tags_info.tag_size and idx < self.tags_info.s_tag_start + self.s:
					re[idx] = self.need
					idx += 1

				idx = self.tags_info.p_tag_start
				while idx < self.tags_info.x_tag_start and idx < self.tags_info.p_tag_start + self.p_max:
					re[idx] = self.need
					idx += 1
				#while idx < self.tags_info.p_tag_start + self.max_p and idx < self.tags_info.p_tag_start + self.p:
				#	re[idx] = self.need
				#	idx += 1

			if self.stack[-1] == 15:
				pass
			else:
				re[self.prev_variable] = self.mask
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[1] = self.need
			return re
		
	def update(self, ix):
		if ix < self.tags_info.tag_size:
			if ix == 1:
				pass
			if ix >= 5 and ix < self.tags_info.k_rel_start:
				self.stack.append(ix)
				if ix == 5:
					self.stack_ex.append([])
					self.stack_drs.append(ix)
				elif ix == 6:
					self.stack_drs.append(ix)
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = -1
			elif ix >= self.tags_info.k_rel_start and ix < self.tags_info.p_rel_start:
				self.stack.append(ix)
				self.stack_ex[-1].append(ix - self.tags_info.k_rel_start + self.tags_info.k_tag_start)
				#self.k += 1
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = -1

			elif ix >= self.tags_info.p_rel_start and ix < self.tags_info.k_tag_start:
				self.stack.append(ix)
				#self.p += 1
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = -1
			elif ix == 4:
				if self.stack[-1] == 5:
					self.stack_ex.pop()
					self.stack_drs.pop()
				elif self.stack[-1] == 6:
					self.stack_drs.pop()
				self.stack.pop()
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = -1
			else:

				if ix >= self.tags_info.k_tag_start and ix < self.tags_info.p_tag_start:
					#assert self.k >= ix - self.tags_info.k_tag_start + 1
					#if self.k == ix - self.tags_info.k_tag_start + 1:
					#	self.k += 1
					pass
				elif ix >= self.tags_info.p_tag_start and ix < self.tags_info.x_tag_start:
					#assert self.p >= ix - self.tags_info.p_tag_start + 1
					#if self.p == ix - self.tags_info.p_tag_start + 1:
					#	self.p += 1
					pass
				elif ix >= self.tags_info.x_tag_start and ix < self.tags_info.e_tag_start:
					assert self.x >= ix - self.tags_info.x_tag_start + 1
					if self.x == ix - self.tags_info.x_tag_start + 1:
						self.x += 1
				elif ix >= self.tags_info.e_tag_start and ix < self.tags_info.s_tag_start:
					assert self.e >= ix - self.tags_info.e_tag_start + 1
					if self.e == ix - self.tags_info.e_tag_start + 1:
						self.e += 1
				elif ix >= self.tags_info.s_tag_start and ix < self.tags_info.tag_size:
					assert self.s >= ix - self.tags_info.s_tag_start + 1
					if self.s == ix - self.tags_info.s_tag_start + 1:
						self.s += 1
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = ix

		else:
			self.stack.append(ix)
			self.prev_prev_variable = self.prev_variable
			self.prev_variable = -1

	def _print_state(self):
		print "stack", self.stack
		print "stack_ex", self.stack_ex
		print "stack_drs", self.stack_drs
		print "k_use", self.k_use
		print "user_k", self.user_k
#		print "kpxes", self.k, self.p, self.x, self.e, self.s
		print "xes", self.x, self.e, self.s
	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]

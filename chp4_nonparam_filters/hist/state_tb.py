import numpy as np
import pandas as pd

class StateTable(object):
    """
    Discretize the state space. 
    """
    def __init__(self, interval, num_grids, names):
        """
        interval is a list of tuples where each tuple specifies the min and max of every state dimension. 
        num_grids is a list where each element specifies the number discretized states for that dimension.
        names is an optional arg which specifies the name of each state dimension. 
        """
        self.n_state = np.prod(num_grids)
        self.interval = np.array(interval) 
        self.num_grids = np.array(num_grids)
        self.quantization = np.array([(i[1] - i[0]) / float(g)  for i, g in zip(interval, num_grids)])
        self.names = names
        self.build_tb()

    def build_tb(self):
        """
        Build the discretized state space table. 
        Table is a dictionary of {s_id : [(min_v1, max_v1), (min_v2, max_v2), ...]}
        """
        self.tb = {}
        # header for the table
        self.tb[-1] = ('s_id', *self.names)
        for s_id, indices in enumerate(np.ndindex(*self.num_grids)):
            value_range = []
            for dim_idx, idx in enumerate(indices):
                dim_min = self.interval[dim_idx][0]
                delta = self.quantization[dim_idx]
                range_min = dim_min + delta * idx 
                range_max = dim_min + delta * (idx + 1) 
                value_range.append((range_min, range_max))
            self.tb[s_id] = value_range
    
    def s_id(self, x):
        """
        Given a continuous variable x = x, query the state.
        """
        x = x.flatten()
        min_values = np.array([v[0] for v in self.interval])
        bins = np.clip(np.ceil((x -min_values)/ self.quantization) - 1, a_min=0, a_max=None)
        s_id = 0
        inverse_num_grids = self.num_grids[::-1]
        for i in range(len(inverse_num_grids)):
            s_id += bins[i] * np.prod(inverse_num_grids[0 : len(inverse_num_grids) - i - 1])
        return int(s_id)

    def s_range(self, s):
        """ Return the value range of the given state """
        return self.tb[s] 

    def value(self, s):
        """
        Given a discrete state s = s, query the continuous value.
        """
        values = []
        value_range = self.tb[s]
        for _range in value_range:
            values.append((_range[1] - _range[0])/2.0 + _range[0])
        return np.array(values) 

    def all_states(self):
        """
        Get all the discrete state.
        """
        return np.arange(0, self.n_state)

    def all_values(self):
        """
        Get all the discrete state.
        """
        states = self.all_states()
        values = []
        for s in states:
            values.append(self.value(s))
        return np.array(values)
import heapq
from collections import Counter

class Node:
    def __init__(self, key, freq):
        self.key = key
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq
    
    def __gt__(self, other):
        return self.freq > other.freq
    
    def __eq__(self, other):
        return self.freq == other.freq

class Tree:
    def __init__(self, walk):
        self.walk = walk
        self.freq = None
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.root = None

    def build(self):
        frequency = self.get_freq()
        self._make_heap(frequency)
        self._merge_nodes()
        self._make_codes()

    def _get_freq(self):
        self.freq = dict(Counter(self.walk))

    def _make_heap(self, frequency):
        for key in frequency:
            node = Node(key, frequency[key])
            heapq.heappush(self.heap, node)

    def _merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            
            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def _make_codes_helper(self, root, current_code):
        if(root is None):
            return None
        if(root.key is not None):
            self.codes[root.key] = current_code
            self.reverse_mapping[current_code] = root.key
            return None

        self._make_codes_helper(root.left, current_code + "0")
        self._make_codes_helper(root.right, current_code + "1")
        
    def _make_codes(self):
        root = heapq.heappop(self.heap)
        self.root = root
        current_code = ""
        self._make_codes_helper(root, current_code)

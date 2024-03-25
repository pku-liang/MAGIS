from typing import Dict, TypeVar

_VT = TypeVar("_VT")


class UnionFindSet(Dict[_VT, _VT]):
    def find_root(self, nd: _VT) -> _VT:
        prt = self[nd]
        if prt is None or isinstance(prt, (set, list)):
            return nd
        root = self.find_root(prt)
        self[nd] = root
        return root

    def union_trees(self, a: _VT, b: _VT):
        ra, rb = self.find_root(a), self.find_root(b)
        if ra == rb:
            return
        self[ra] = rb

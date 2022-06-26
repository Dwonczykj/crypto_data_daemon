
from typing import Any, Callable, TypeVar


T = TypeVar('T')

def ifcall(X:T, predicateCallX:Callable[[T],Any],callIfWithX:Callable[[T],Any]):
    if predicateCallX(X):
        return callIfWithX(X)
    
    
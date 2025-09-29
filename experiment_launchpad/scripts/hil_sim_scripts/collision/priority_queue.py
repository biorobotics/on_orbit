'''
Provides functions that implement a numba-compatible min-heap based priority queue
'''
from typing import TypeVar,Tuple,NamedTuple,List
import heapq
from numba import njit
from collections import namedtuple


Data=TypeVar('Data')
Priority=TypeVar('Priority')
Heap=List[Tuple[Priority,int,Data]]
class PriorityQueue(NamedTuple):
    heap:Heap
    counter:int
    entries:int

def pymake_queue(first_item:Data,first_priority:Priority):
    '''
    create a (numba-compatible) min-heap priority queue with one element

    Parameters: first_item : Any
                    the first element to put into the queue
                first_priority : comparable, usually float
                    priority of first element. Min heap, so smaller priority goes to the front
    Return:     queue : PriorityQueue
                    namedtuple storing the heap, the unique index assigned to the next item to be added, and the current # of elements
    '''
    heap=[(first_priority,0,first_item)]
    heapq.heapify(heap)
    counter=1
    return PriorityQueue(heap,counter,1)
make_queue=njit(pymake_queue)

def pyadd_item(pq:PriorityQueue,item:Data,priority:Priority):
    '''
    add an element to the PriorityQueue. The old PriorityQueue object is no longer usable!

    Parameters: pq : PriorityQueue
                    the old PriorityQueue. pq.heap gets modified, but pq.counter and pq.entries do not update
                item : Any
                    element to put into the queue
                priority : comparable, usually float
                    priority of the item. Smaller goes to front.
    Return:     queue : PriorityQueue
                    namedtuple storing the heap
    '''
    heapq.heappush(pq.heap,(priority,pq.counter,item))
    return PriorityQueue(pq.heap,pq.counter+1,pq.entries+1)
add_item=njit(pyadd_item)

def pyget_item(pq:PriorityQueue)->Tuple[PriorityQueue,Priority,Data]:
    '''
    get the item in the queue with the smallest priority, removing it from the queue in the process

    Parameters: pq : PriorityQueue
                    the old PriorityQueue. pq.heap gets modified, but pq.counter and pq.entries do not update
    Return:     queue : PriorityQueue
                    modified queue
                priority : comparable, usually float
                    priority of the item that was removed
                item : Any
                    element that was removed
    '''
    priority,_,item=heapq.heappop(pq.heap)
    return PriorityQueue(pq.heap,pq.counter,pq.entries-1),priority,item
get_item=njit(pyget_item)

def pyis_empty(pq:PriorityQueue)->bool:
    '''
    check if the queue is empty

    Parameters: pq : PriorityQueue
                    the queue to check. NOT mutated.
    Return:     is_empty : bool
                    true if there are no entries left in the queue
    '''
    return pq.entries==0
is_empty=njit(pyis_empty)
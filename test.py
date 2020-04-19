# coding: utf-8
from contexttimer import Timer
from numba import jit


@jit(nopython=True, cache=True)
def jit_test():
    a = 0
    for i in range(10 ** 8):
        a += 1

def base_test():
    with Timer() as t:
        a = 0
        for i in range(10 ** 8):
            a += 1
        print(t.elapsed)

with Timer() as t:
    jit_test()
    print(t.elapsed)
base_test()

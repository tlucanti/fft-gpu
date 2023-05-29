
import numpy as np
import random
from threading import Thread, Barrier
from concurrent.futures import ThreadPoolExecutor

def _is_power_of_2(x):
    return (x & (x - 1)) == 0

def _print(x):

    def __pr(n):
        if abs(n) < 1e-3:
            return '0'
        else:
            return f'{n:g}'

    for i in x:
        if abs(i.imag) < 1e-3:
            print(__pr(i.real), end=' ')
        else:
            print(end=f'({__pr(i.real)}, {__pr(i.imag)}) ')
    print()


def lib_fft(x):
    assert _is_power_of_2(len(x))

    return np.fft.fft(x)

def random_range(*args):
    a = list(range(*args))
    random.shuffle(a)
    return a

def fft_parallel(x):
    assert _is_power_of_2(len(x))


    def _fft_parallel(x, out, i, barrier):
        N = len(x)
        k = N
        thetaT = 3.14159265358979323846264338328 / N
        phiT = np.cos(thetaT) - 1j * np.sin(thetaT)

        while k > 1:
            n = k
            k >>= 1
            phiT = phiT * phiT
            T = 1.0

            for l in range(k):
                a = i
                if a in range(l, N, n):
                    b = a + k
                    t = x[a] - x[b]
                    out[a] = x[a] + x[b]
                    out[b] = t * T
                T *= phiT
            barrier.wait()
            #if (i == 0):
            #   print(k, end=' : ')
            #    _print(out)
            out, x = x, out

        m = int(np.log2(N))

        a = i
        if 1:
            b = a

            # Reverse bits
            b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1))
            b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2))
            b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4))
            b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8))
            b = ((b >> 16) | (b << 16)) >> (32 - m)
            if b > a:
                t = x[a]
                x[a] = x[b]
                x[b] = t

    threads = []
    barrier = Barrier(len(x))
    out = np.zeros_like(x)

    for i in range(len(x)):
        args = (x, out, i, barrier)
        t = Thread(target=_fft_parallel, args=args)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def fast_fft(x):
    assert _is_power_of_2(len(x))

    out = np.zeros_like(x)
    N = len(x)
    k = N
    thetaT = 3.14159265358979323846264338328 / N
    phiT = np.cos(thetaT) - 1j * np.sin(thetaT)
    swap = 0
    iter = 0

    while k > 1:
        n = k
        k >>= 1
        phiT = phiT * phiT
        T = 1.0

        for l in range(k):
            iter += 1
            print(l, N, n)
            for a in random_range(l, N, n):
                b = a + k
                t = x[a] - x[b]
                out[a] = x[a] + x[b]
                out[b] = t * T
            T *= phiT
        swap += 1
        out, x = x, out

    print(iter)

    # Decimate
    m = int(np.log2(N))
    print('swaps', swap)
    if swap % 2:
        out, x = x, out
        swap += 1
    print('swaps', swap)

    for a in range(N):
        b = a

        # Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1))
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2))
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4))
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8))
        b = ((b >> 16) | (b << 16)) >> (32 - m)
        if b > a:
            t = x[a]
            x[a] = x[b]
            x[b] = t

def diff(a, b):
    assert len(a) == len(b)

    size = len(a)
    for i in range(size):
        d = abs(a[i] - b[i]) / abs(a[i] + b[i])
        if d > 1e-3:
            raise ValueError
    print('diff: OK')

def test():
    size = 2**8
    a = np.random.rand(size).astype(complex)
    b = np.copy(a)
    c = np.copy(a)

    a = lib_fft(a)
    _print(a)
    #fast_fft(b)
    #fft_parallel(c)
    fft_parallel(b)
    _print(b)
    diff(a, b)

    #_print(a)
    #_print(b)
    #_print(c)

def run():
    size = 2**10
    a = np.arange(1, size + 1).astype(complex)
    b = lib_fft(a)
    fft_parallel(a)
    print(end='res : ')
    _print(a)
    print(end='lib : ')
    _print(b)
    diff(a, b)

if __name__ == '__main__':
    run()

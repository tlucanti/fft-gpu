
import numpy as np
import random
from threading import Thread, Barrier
from concurrent.futures import ThreadPoolExecutor

def _is_power_of_2(x):
    return (x & (x - 1)) == 0

def _print(x):
    for i in x:
        print(end=f'{i:.2f} ')
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

    while k > 1:
        n = k
        k >>= 1
        phiT = phiT * phiT
        T = 1.0

        for l in range(k):
            for a in random_range(l, N, n):
                b = a + k
                t = x[a] - x[b]
                out[a] = x[a] + x[b]
                out[b] = t * T
            T *= phiT
        swap += 1
        out, x = x, out

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


def test():
    size = 2**12
    a = np.random.rand(size).astype(complex)
    b = np.copy(a)

    a = lib_fft(a)
    fft_parallel(b)

    #_print(a)
    #_print(b)
    for i in range(size):
        d = (a[i] - b[i]) / (a[i] + b[i])
        if d > 1e-3:
            raise ValueError

def main():
    test()

if __name__ == '__main__':
    main()


import numpy as np
import random

def _is_power_of_2(x):
    return (x & (x - 1)) == 0

def _print(x):
    for i in x:
        print(end=f'{i:.2f} ')
    print()


def lib_fft(x):
    assert _is_power_of_2(len(x))

    return np.fft.fft(x)

def my_fft_rec_np(x):
    assert _is_power_of_2(len(x))

    def _fft(x):
        N = len(x)

        if N == 1:
            return x
        else:
            X_even = _fft(x[::2])
            X_odd = _fft(x[1::2])
            factor = np.exp(-2j*np.pi*np.arange(N)/ N)

            X = np.concatenate([
                X_even+factor[:N // 2] * X_odd,
                 X_even+factor[N // 2:] * X_odd
            ])
        return X

    return _fft(x)

def my_fft_rec(x):
    assert _is_power_of_2(len(x))

    def _fft(x):
        N = len(x)

        if N == 1:
            return x
        else:
            x[::2] = _fft(x[::2])
            x[1::2] = _fft(x[1::2])
            factor = np.exp(-2j*np.pi*np.arange(N)/ N)

            x = np.concatenate([
                x[::2] + factor[:N // 2] * x[1::2],
                x[::2] + factor[N // 2:] * x[1::2]
            ])
        return x

    return _fft(x)

def my_fft_rec_inplace(x):
    assert _is_power_of_2(len(x))

    def _fft(x, N, start, end, step):
        if N == 1:
            return
        else:
            _fft(x, N // 2, start, end, step * 2)
            _fft(x, N // 2, start + 1, end, step * 2)

            for k in range(N // 2):
                p = x[k]
                q = np.exp(-2j * np.pi * k / N) * x[k + N // 2]
                x[k] = p + q
                x[k + N // 2] = p - q
        return

    _fft(x, len(x), 0, len(x), 1)
    return x

def main():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    lib = lib_fft(x)
    my = my_fft_rec_inplace(x)
    _print(lib)
    _print(my)
    assert abs(sum(lib - my)) < 1e-5

if __name__ == '__main__':
    main()

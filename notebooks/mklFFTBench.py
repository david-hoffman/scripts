# References:
#
# http://software.intel.com/en-us/intel-mkl

import time

import numpy
import numpy.fft.fftpack

import pyfftw

pyfftw.interfaces.cache.enable()

import matplotlib.pyplot as plt
from matplotlib import ticker


def show_info():
    try:
        import mkl

        print("MKL MAX THREADS:", mkl.get_max_threads())
    except ImportError:
        print("MKL NOT INSTALLED")


def plot_results(datas, factor=None, algo="FFT"):
    xlabel = r"Array Size (2^x)"
    ylabel = "Speed (GFLOPs)"
    backends = ["numpy+mkl", "numpy", "pyFFTW"]

    plt.clf()
    fig1, ax1 = plt.subplots()
    plt.figtext(0.90, 0.94, "Note: higher is better", va="top", ha="right")
    w, h = fig1.get_size_inches()
    fig1.set_size_inches(w * 1.5, h)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.get_xaxis().set_minor_locator(ticker.NullLocator())
    ax1.set_xticks(datas[0][:, 0])
    ax1.grid(color="lightgrey", linestyle="--", linewidth=1, alpha=0.5)
    if factor:
        ax1.set_xticklabels([str(int(x)) for x in datas[0][:, 0] / factor])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(datas[0][0, 0] * 0.9, datas[0][-1, 0] * 1.025)
    plt.suptitle("%s Performance" % ("FFT"), fontsize=28)

    for backend, data in zip(backends, datas):
        N = data[:, 0]
        plt.plot(N, data[:, 1], "o-", linewidth=2, markersize=5, label=backend)
        plt.legend(loc="upper left", fontsize=18)

    plt.savefig(algo + ".png")


# assumes mkl FFTs are run before nonMKL FFTS
def run(a, repeat, size, meth="mkl", mkloutput=[]):
    # print(len(mkloutput))
    bs = []
    # build the cache
    if meth == "pyfftw":
        b = pyfftw.interfaces.numpy_fft.fftn(a)
    start_time = time.time()
    for i in range(repeat):
        if meth == "mkl":
            b = numpy.fft.fftn(a)
        elif meth == "numpy":
            b = numpy.fft.fftpack.fftn(a)
        elif meth == "pyfftw":
            b = pyfftw.interfaces.numpy_fft.fftn(a)
        bs.append(b)
    if meth != "mkl":
        assert len(bs) == len(mkloutput)
    time_taken = time.time() - start_time
    return time_taken, bs


def main():
    show_info()
    dataMKL = []
    dataNumpy = []
    datapyFFTW = []
    print(
        "\n%8s %8s %16s %16s %16s"
        % ("trials", "2^n", "time(s) MKL", "time(s) No MKL", "time(s) pyFFTW")
    )
    print("----------------------------" * 3)
    for n in range(4, 25):
        size = 2 ** n
        # to keep the experiment from taking too long
        if n < 10:
            trials = 1000
        elif n < 20:
            trials = 100
        else:
            trials = 10
        mflop = 5.0 * size * numpy.log2(size)
        gflop = mflop / 1000
        a = numpy.random.randn(size) + 1j * numpy.random.randn(size)
        a = a.astype(numpy.complex64)
        # MKL
        s, output = run(a, trials, size)
        avg_ms = (s / trials) * 1000000
        dataMKL.append(numpy.asarray([n, gflop / avg_ms]))
        # no MKL
        s2, output2 = run(a, trials, size, meth="numpy", mkloutput=output)
        avg_ms = (s2 / trials) * 1000000
        dataNumpy.append(numpy.asarray([n, gflop / avg_ms]))
        # pyFFTW
        s3, output3 = run(a, trials, size, meth="pyfftw", mkloutput=output)
        avg_ms = (s3 / trials) * 1000000
        datapyFFTW.append(numpy.asarray([n, gflop / avg_ms]))
        print("%8i %8i %12.4fs %12.4fs %12.4fs" % (trials, n, s, s2, s3))
    datas = numpy.asarray(
        [numpy.asarray(dataMKL), numpy.asarray(dataNumpy), numpy.asarray(datapyFFTW)]
    )
    plot_results(datas, algo="FFT")


if __name__ == "__main__":
    main()

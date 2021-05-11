import cProfile
import pstats
import io


def easyProfiler(func):
    """ to profile any function decorate it with this profiler.
    Profiling result will be dumped on the console.
    """
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortBy = "tottime"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortBy)
        ps.print_stats(.1)
        print(s.getvalue())
        return retval
    return inner

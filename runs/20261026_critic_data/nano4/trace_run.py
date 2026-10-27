"""Run collect.py with a Python stack dump every 30 s on stderr (diagnosing a worker that makes no progress).
usage: python trace_run.py <collect.py args...>"""
import faulthandler, os, runpy, sys
faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(here, "..", "collect.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

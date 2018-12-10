import time

class TimeLeft(object):

    def __init__(self, n, n_0=0):
        """Calculates the time left between the current iteration i and the total 
        number of iterations n while the calculation starts at n_0. 
        So for n_0=5, n=10, and i=7 it calculates how long the remaining 3
        iterations will take based on the time passed between the initialization
        and now given there were 2 iterations in that time.
        
        Arguments:
            n {int} -- number of iterations
        
        Keyword Arguments:
            n_0 {int} -- starting number (default: {0})
        """

        self.max_iterations = n
        self.start_iteration = n_0
        self.start_time = time.time()
        self.lap_time = self.start_time
        self.counter = n_0

    def reset(self):
        self.start_time = time.time()
    
    def time_left(self, ii):
        return time_left(self.start_time, ii-self.start_iteration, self.max_iterations-self.start_iteration)
    
    def pretty_time_left(self, ii):
        return pretty_time_left(self.start_time, ii-self.start_iteration, self.max_iterations-self.start_iteration)

    def elapsed(self):
        return pretty_print_time(time.time()-self.start_time)

    def lap(self):
        dt = time.time() - self.lap_time
        self.lap_time = time.time()
        return pretty_print_time(dt)

    def __str__(self):
        val = self.pretty_time_left(self.counter-self.start_iteration)
        return val

    def __format__(self, format_spec):
        return str(self)

    def __call__(self, counter):
        self.counter = counter
        return self

    def print(self, counter):
        print('\r{}        '.format(self(counter)), end='', flush=True)

def pretty_time_left( start_time, current_iteration, max_iterations ):
    return pretty_print_time( time_left( start_time, current_iteration, max_iterations ) )

def pretty_print_time( remaining ):
    if remaining < 60:
        time_string = "{:.0f}s".format(remaining)
    elif remaining < 3600:
        time_string = "{}:{:02d} min".format(int(remaining/60), int(remaining%60))
    elif remaining < 86400:
        time_string = "{}:{:02d}:{:02d} h".format(int(remaining/3600), int(remaining%3600/60), int(remaining%60))
    else:
        time_string = "{}d {}:{:02d}:{:02d} h".format(int(remaining/86400), int(remaining%86400/3600), int(remaining%3600/60), int(remaining%60))
    return time_string

def time_left( start_time, current_iteration, max_iterations ):
    dt = time.time() - start_time
    if current_iteration != 0:
        return (dt / current_iteration) * (max_iterations - current_iteration)
    else:
        print('Iteration 0 not supported. Did you forget to add 1?')
        return None    

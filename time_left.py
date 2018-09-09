import time

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
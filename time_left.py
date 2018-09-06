import time

def pretty_time_left( start_time, current_iteration, max_iterations ):
    remaining = time_left( start_time, current_iteration, max_iterations )
    if remaining < 60:
        time_string = "{.0f}s".format(remaining)
    elif remaining < 3600:
        time_string = "{}:{} min".format(int(remaining/60), int(remaining%60))
    elif remaining < 86400:
        time_string = "{}:{}:{} h".format(int(remaining/3600), int(remaining%3600/60), int(remaining%60))
    else:
        time_string = "{}d {}:{}:{} h".format(int(remaining/86400), int(remaining%86400/3600), int(remaining%3600/60), int(remaining%60))
    return time_string

def time_left( start_time, current_iteration, max_iterations ):
    dt = time.time() - start_time
    if current_iteration != 0:
        return (dt / current_iteration) * (max_iterations - current_iteration)
    else:
        return np.nan    
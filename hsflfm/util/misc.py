from datetime import datetime

def get_timestamp():
    str_format = "%Y%m%d_%H%M%S" 
    time_string = datetime.now().strftime(str_format) 
    return time_string
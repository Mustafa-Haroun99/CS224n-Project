from datetime import datetime

def generate_experiment_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")
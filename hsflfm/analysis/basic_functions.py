import torch

# displacements should be (# points, # frames)
# this will just return the point with the average highest velocity
def get_strike_center(displacements):   
    derivs = torch.diff(displacements, axis=1)
    avg = torch.mean(torch.abs(derivs), axis=0) 
    strike_point = torch.argmax(avg) 
    return strike_point

# displacements should be (# points, # frames)
# this will return the index at which the displacements first peak
# after the strike center
def get_peak_indices(displacements):
    strike_center = get_strike_center(displacements)
    derivs = torch.diff(displacements, axis=1)
    peak_indices = torch.zeros(displacements.shape[0], dtype=torch.uint16)

    for i in range(len(displacements)):
        line_d = derivs[i, strike_center:]
        for index, value in enumerate(line_d):
            if torch.sign(value) != torch.sign(line_d[0]):
                stop_point = index + strike_center
                break
        peak_indices[i] = stop_point
    return peak_indices
def generate_path(pseudo_id):
    """ Generates path with the following structure: ps/eu/doid """
    pseudo_id = str(pseudo_id)
    return pseudo_id[0:2] + '/' + pseudo_id[2:4] + '/' + pseudo_id[4:-1]
  

def receptive_field(n, kernel_size, n_repeat=1):
    if n < 1:
      return 0
    if n == 1 and n_repeat > 1:
      return kernel_size + (n_repeat-1)*((kernel_size-1) * 2**(n-1))
    if n == 1:
      return kernel_size
    prev = receptive_field(n-1, kernel_size)
    return prev + n_repeat*((kernel_size-1) * 2**(n-1))
  
# generates valid values for input x output in the u-net architecture
# each convolution pass decrease the network size by 2 voxels (but it is applied twice, reducing in fact by 4 voxels)

def apply_convolution(valid_values):
    if (valid_values[-1] - 4 < 0):
        return False
    size = valid_values[-1] - 4
    valid_values.append(size)
#    print(" new size - convolution: {}".format(size))
    return True

    
def downsample(downsampling_value, valid_values):
    if (valid_values[-1] == 0):
        return False
    
#    print("downsample verification - size: {} - factor: {} - result {}".format(valid_values[-1], downsampling_value, valid_values[-1]%downsampling_value))
    if (valid_values[-1] % downsampling_value != 0):
        return False
    
    size = valid_values[-1] / downsampling_value
    valid_values.append(size)
#    print(" new size - downsample: {}".format(size))
    return True


def upsample(downsampling_value, valid_values):
    size = valid_values[-1] * downsampling_value
#    print(" new size - upsample: {}".format(size))
    valid_values.append(size)


def input_output(input_size, downsampling_values, valid_values):
    number_of_steps = len(downsampling_values)
    # downsampling
    for i in range(number_of_steps):
        if (apply_convolution(valid_values) == False):
            return False
        if (downsample(downsampling_values[i], valid_values) == False):
            return False
        
    if (apply_convolution(valid_values) == False):
        return False
    
    # upsampling
    for i in range(number_of_steps):
        upsample(downsampling_values[number_of_steps - i - 1], valid_values)
        if (apply_convolution(valid_values) == False):
            return False
        
    return True


if __name__ == '__main__':
    downsampling_values = [3, 3, 3]
    valid_values = []
    min_x = 0
    
    for input_size in range(500):
        valid_values.append(input_size)
        success = input_output(input_size, downsampling_values, valid_values)
        if (success == False):
            valid_values = []
            continue
        else:
            print("Configuration for x >= {}, downsampling {} :". format(min_x, downsampling_values))
            print(valid_values)
            print(", context {}".format(valid_values[0] - valid_values[-1]))
            min_x = input_size + 1

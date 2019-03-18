def calc_output_and_feature_size(height, width):
    output = (height/(2**3))*(width/(2**3))
    features = ((height/2**1)*(width/2**1)) * 64 + \
               ((height/2**2)*(width/2**2)) * 128 + \
               ((height/2**3)*(width/2**3)) * 256 + \
               ((height/2**3)*(width/2**3)) * 512
    return int(output), int(features)

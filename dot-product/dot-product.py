import numpy as np

def dot_product(x, y):
    vec_x = np.array(x)
    vec_y = np.array(y)
    
    return np.dot(vec_x, vec_y)

v1 = [1, 2, 3]
v2 = [4, 5, 6]

sonuc = dot_product(v1, v2)
print("Sonuç:", sonuc)
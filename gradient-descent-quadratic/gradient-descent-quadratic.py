def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = x0
    
    for _ in range(steps):
        derivative = 2 * a * x + b
        x = x - lr * derivative
        
    return float(x)

output1 = gradient_descent_quadratic(a=1, b=-4, c=3, x0=0, lr=0.1, steps=50)
print(f"Örnek 1 Çıktısı: {output1:.1f}")  

output2 = gradient_descent_quadratic(a=0.5, b=-1, c=0, x0=-5, lr=0.2, steps=100)
print(f"Örnek 2 Çıktısı: {output2:.1f}")  
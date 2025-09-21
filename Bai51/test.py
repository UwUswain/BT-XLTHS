# problem51_inverse.py
import numpy as np

def compute_b(a, K=10):
    """
    Tính dãy b_k từ dãy a_k theo quan hệ:
    sum_{m=0}^k a_m b_{k-m} = delta[k]
    
    a: list/array chứa a_0, a_1, ...
    K: số hệ số b muốn tính
    """
    a = np.array(a, dtype=float)
    b = np.zeros(K)
    b[0] = 1.0 / a[0]  # từ điều kiện k=0
    
    for k in range(1, K):
        s = 0
        for m in range(1, min(len(a), k+1)):
            s += a[m] * b[k-m]
        b[k] = -s / a[0]
    return b

if __name__ == "__main__":
    # (c) Với a0=1, a1=0.5, a2=0.25, còn lại bằng 0
    a = [1, 0.5, 0.25]
    b = compute_b(a, K=10)

    print("Các hệ số a_k:", a)
    print("Các hệ số b_k (nghịch đảo):")
    for i, val in enumerate(b):
        print(f"b[{i}] = {val:.5f}")

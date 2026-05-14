import numpy as np
import random
from src.utils import set_seed
'''
    테스트 방법:
    터미널에 python -m tests.seed_test 입력
'''

def generate_random_values():
    fruit = ['apple', 'banana', 'pear', 'strawberry', 'pineapple']
    
    random.shuffle(fruit)
    rand_float = random.uniform(10.5, 20.5)
    np_array = np.random.rand(5)

    return fruit, rand_float, np_array


def test_seed_reproducibility():
    # 첫 번째 실행
    set_seed(42)
    result1 = generate_random_values()

    # 두 번째 실행
    set_seed(42)
    result2 = generate_random_values()

    # 비교 (값이 다르면 AssertionError)
    assert result1[0] == result2[0], "shuffle 결과 다름"
    assert result1[1] == result2[1], "random.uniform 결과 다름"
    assert np.allclose(result1[2], result2[2]), "numpy 결과 다름"

    print("seed 고정 정상 작동")

# python tests/seed_test.py 직접 실행 시 True (테스트 코드 자동 실행 방지)
if __name__ == "__main__":
    test_seed_reproducibility()
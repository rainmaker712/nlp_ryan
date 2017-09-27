"""
울면서 배우는 텐서플로우
http://chanacademy.tistory.com/category/%EC%9A%B8%EB%A9%B4%EC%84%9C%20TF

텐서플로
1. 값 연산 정의
2. 세션 열기
3. 세션 연산 실행
"""

import tensorflow as tf

const = tf.constant("hello world")

sess = tf.Session()

hello = sess.run(const)
print(hello)

sess.close()


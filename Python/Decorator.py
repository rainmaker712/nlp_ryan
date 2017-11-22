#11월 20일, Unit 44 데코레이터
#https://dojang.io/mod/page/view.php?id=1131

#데코레이터는 함수를 수정하지 않은 상태에서 추가 기능을 구현하기 위해 사용

class Calc:
    @staticmethod
    def add(a,b):
        print(a,b)

#함수의 시작과 끝을 출력하는 데코레이터
def trace(func):
    def wrapper():
        print(func.__name__, '함수 시작')
        func()
        print(func.__name__, '함수 끝')
    return wrapper

@trace
def hello():
    print('hello')

@trace
def world():
    print('world')

# trace_hello = trace(hello) #데코레이터에 호출할 함수 넣기
# trace_hello() #반환된 함수를 호출
# trace_world = trace(world)
# trace_world()

hello()
world()
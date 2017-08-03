from enum import Enum
import random

class SayHelloType(Enum):
    GENERAL = 0
    FIRST_VISIT = 1
    REVISIT = 2
    WHEN_TIME_MORNING = 3
    WHEN_TIME_NIGHT = 4
    
say_hello_templates = {
    SayHelloType.GENERAL: ('<NAME>', ('ㅎㅇ', '안녕하세요?', '오~ 안녕!!', '인사성 좋네')),
    SayHelloType.FIRST_VISIT: (('반가워', '처음뵙겠습니다'), '<NAME>'),
    SayHelloType.REVISIT: ('<NAME>', ('오오 안녕안녕?', '잘 지냈어?')),
    SayHelloType.WHEN_TIME_MORNING: (('좋은 아침이야', '굿모닝', '일찍일어났네?'), '<NAME>'),
    SayHelloType.WHEN_TIME_NIGHT: (('오늘도 밤이 왔습니다',),),    
}


def say_hello(say_hello_type=None, user_name=None):
    if (say_hello_type == None) or (type(say_hello_type) != SayHelloType):
        say_hello_type = SayHelloType.GENERAL
    
    templates = say_hello_templates[say_hello_type]
    sentence = []
    for term in templates:
        if type(term) == str:
            if (term == '<NAME>') and (user_name is not None):
                sentence.append(user_name)
                continue
        if type(term) == tuple:
            sentence.append(random.choice(term))
            
    return ' '.join(sentence)
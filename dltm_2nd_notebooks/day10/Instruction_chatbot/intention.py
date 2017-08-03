from enum import Enum

class Requests(Enum):
    HELLO = 0
    RECOMMENDE_MOVIE = 1
    SEARCH_MOVIE = 2
    SEARCH_ACTOR = 3
    SEARCH_DIRECTOR = 4
    RESERVE_MOVIE = 5
    

intention_templates = { (('안녕', 'ㅎㅇ', '헬로'),): Requests.HELLO,
                    (('영화', '프로', '거'), ('골라', '추천', '보여줘', '보여')): Requests.RECOMMENDE_MOVIE,
                    ('영화', ('뭐야', '알려줘', '찾아줘')): Requests.SEARCH_MOVIE, 
                    (('나온', '출연한'), ('영화', '프로', '거'), ('뭐야', '알려줘', '찾아줘')): Requests.SEARCH_MOVIE, 
                    ('나온', ('영화', '거', '프로')): Requests.SEARCH_MOVIE, 
                    ('배우', ('누구야', '알려줘', '찾아줘')): Requests.SEARCH_ACTOR, 
                    ('감독', ('누구야', '알려줘', '찾아줘')): Requests.SEARCH_DIRECTOR,
                    ('영화', ('예약', '잡아')): Requests.RESERVE_MOVIE
                  }


def intention_classifier(tokens):
    def has_intention(tokens, terms):
        for term in terms:
            if type(term) == str:
                if (term in tokens) == False:
                    return False
            elif type(term) == tuple:
                has_at_least_one = False
                for alternative in term:
                    if alternative in tokens:
                        has_at_least_one = True
                        break
                if not has_at_least_one:
                    return False
        return True
       
    intentions = []
    for terms, request in intention_templates.items():
        if has_intention(tokens, terms):
            intentions.append(request)
    
    intentions = list(set(intentions))
    return intentions
import pickle
import random
from hello import SayHelloType, say_hello
from intention import Requests, intention_classifier
from ner import named_entity_extractor


class MovieBot:

    def __init__(self, param_fname):
        with open(param_fname, 'rb') as f:
            params = pickle.load(f)

        self.actor2id = params['actor2id']
        self.actor2movie = params['actor2movie']
        self.movie2actor = params['movie2actor']
        self.id2movie = params['id2movie']
        self.id2actor = params['id2actor']
        
        self.dummy_message = ['음.. 무슨 말이에요?', '허허..', '음..', '잘 못알아듣겠어요', '그렇구나']
        self.recommend_movies_message = ['보는거 어때요?', '볼래요?', '추천해요']
        self.fail_movie_search_message = ['잘 못찾겠어요', '아쉽지만 아는 영화가 없네요']
    
    def get_movies_from_actor(self, actor):
        actor_idxs = self.actor2id.get(actor, [])
        if not actor_idxs:
            return [], False        
        movies = []
        for actor_idx in actor_idxs:
            movies += self.actor2movie.get(actor_idx, [])            
        return movies, True

    def recommend_movie_by_actor(self, actor):
        movies, unknown_actor = self.get_movies_from_actor(actor)
        if not unknown_actor:
            return '%s가 누군지 모르겠어요' % actor
        elif not movies:
            return '%s가 나온 영화를 모르겠어요' % actor
        else:
            movie = self.id2movie[random.choice(movies)]
            return '[%s] 이거 %s' % (movie, random.choice(self.recommend_movies_message))
    
    def get_dummy(self):
        return random.choice(self.dummy_message)
    
    def process(self, query, user_name=None, condition=None):
        # Preprocess
        query = query.split()

        # intention classification
        intentions = intention_classifier(query)

        if not intentions:
            return self.get_dummy()

        # intention 마다 다른 action scenario
        if len(intentions) == 1:
            if intentions[0] == Requests.HELLO:
                return say_hello(user_name=user_name)

            if intentions[0] == Requests.SEARCH_MOVIE:
                ners = named_entity_extractor(query)
                for entity, entity_type in ners.items():
                    if entity_type == '<ACTOR>':
                        movies, known_actor = self.get_movies_from_actor(entity)
                        if not movies or not known_actor:
                            return random.choice(self.fail_movie_search_message)
                        else:
                            num_movie = len(movies)
                            movie_names = str([self.id2movie[idx] for idx in movies])
                            return '%d 개의 영화가 있어요. %s' % (num_movie, movie_names)

            return self.get_dummy()

        if (Requests.RECOMMENDE_MOVIE in intentions) and (Requests.SEARCH_MOVIE in intentions):
            ners = named_entity_extractor(query)
            # 여러 명 배우 나오는 건 처리 안함
            for entity, entity_type in ners.items():
                if entity_type == '<ACTOR>':
                    return self.recommend_movie_by_actor(entity)

        return self.get_dummy()
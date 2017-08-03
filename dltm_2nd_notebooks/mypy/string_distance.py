# from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

def levenshtein(s1, s2, cost={}):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    
    def get_cost(c1, c2, cost):
        return 0 if (c1 == c2) else cost.get((c1, c2), 1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + get_cost(c1, c2, cost)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def token_levenshtein(s1, s2, base_levenshtein=levenshtein, debug=False):
    t1 = set(s1.split())
    t2 = set(s2.split())
    
    cost = {(t1_i, t2_j): base_levenshtein(t1_i, t2_j) for t1_i in t1 for t2_j in t2}
    
    if debug: 
        print(cost)
        
    return levenshtein(s1.split(), s2.split(), cost)

class Jamo:
    
    kor_begin = 44032
    kor_end = 55203
    chosung_base = 588
    jungsung_base = 28

    chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
            'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
            'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
            'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
            'ㅡ', 'ㅢ', 'ㅣ']

    jongsung_list = [
        ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
            'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
            'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
            'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
                  'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
                  'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
                  'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        
    def jamo(self, c):
        i = ord(c)
        if not (self.kor_begin <= i <= self.kor_end):
            return [c]
        
        i = i - self.kor_begin
        cho  = i // self.chosung_base
        jung = ( i - cho * self.chosung_base ) // self.jungsung_base 
        jong = ( i - cho * self.chosung_base - jung * self.jungsung_base )

        return [self.chosung_list[cho], self.jungsung_list[jung], self.jongsung_list[jong]]
    
jamo_module = Jamo()

def jamo(c):
    return jamo_module.jamo(c)

def jamo_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    
    def get_jamo_cost(c1, c2):
        return 0 if (c1 == c2) else levenshtein(jamo(c1), jamo(c2))/3

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + get_jamo_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
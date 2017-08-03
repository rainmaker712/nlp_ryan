ner_templates = [
    (('<ACTOR>', ('나온', '출연한', '출연')), 1),
    (('<ACTOR>', ('가', '이'), ('나온', '출연한', '출연')), 2),
    (('<DIRECTOR>', ('만든', '제작', '제작한', '감독', '감독한')), 1),
    (('<DIRECTOR>', ('가', '이'), ('만든', '제작', '제작한', '감독', '감독한')), 2)
]

def named_entity_extractor(tokens):
    def match(tokens, template):
        template_pattern = template[0]
        template_begin = template[1]
        template_end = len(template_pattern) - template_begin
        
        if len(tokens) < template_begin:
            return {}
        
        matched_entities = {}
        for b in range(template_begin, (len(tokens) - template_end + 1)):
            e = b + template_end
            subtokens = tokens[b - template_begin: e]
            
            matched = True
            named_entity = None
            entity_type = None
            for token, template_term in zip(subtokens, template_pattern):
                if (template_term[0] == '<') and (template_term[-1] == '>'):
                    named_entity = token
                    entity_type = template_term
                    continue
                if (token in template_term) == False:
                    matched = False
                    break
            
            if (matched) and (named_entity is not None) and (entity_type is not None):
                matched_entities[named_entity] = entity_type
                
        return matched_entities
    
    matched_entities = {}
    for template in ner_templates:
        matched_entities.update(match(tokens, template))
    
    return matched_entities
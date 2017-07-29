#Binary search

alist = [2,4,6,8,10,14,21]

exp_val = 14

#hint1
low = 0 
high = len(alist) - 1
mid = (low + high) // 2
guess = alist[mid]

#hint2
if guess < item:
    low = mid + 1

#hint3: cannot solve
def binarySearch(list, item):
    low = 0
    high = len(list) - 1
    
    while low <= high: # Key point
        mid = (low + high) // 2
        guess = list[mid]
        if guess == item:
            return mid
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None
        
binarySearch(alist, exp_val)
        
    
    
#my code   
i = 0
while i > 100:
    i += 1
    if guess < exp_val:
        mid = (mid + high) // 2
        guess = alist[mid]
    elif guess > exp_val:
        mid = (mid + low) // 2
        guess = alist[mid]
    else:
        print("Value Location {}, Value {}".format(mid, guess))
        break

    
    

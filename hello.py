def count_substring(string, sub_string):

    if len(string) < len(sub_string):
        return 0
    else:   
        count = 0
        i, j = 0, 0
        while i < len(string):
            
            if string[i] == sub_string[j]:
            
                j += 1
                print(i)
                if j == len(sub_string):
                    count += 1
                    j = 0
                    i -= 1
            else:
                j = 0
            i += 1
    return count

if __name__ == '__main__':
    string = 'WoW!ItSCoOWoWW'
    sub_string = 'oW'
    
    count = count_substring(string, sub_string)
    print(count)
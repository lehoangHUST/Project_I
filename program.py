from collections import Counter

list = [1, 5, 3, 9, 8]

list_color = ['yellow', 'red', 'red', 'yellow', 'blue']

predict = max(list_color, key=list_color.count)
print(predict)
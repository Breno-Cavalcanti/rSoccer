# convert a string of the form '['['all', 'true']']' into [['all', 'true']]
# its a list of lists of strings

def converter(string):
    if string[0] != '[':
        return string
    else:
        string = string[1:-1]
        string = string.split(',')
        string = [converter(x) for x in string]
        return string


# group the elemtns of a list into a list of 2
# example: ['a', 'b', 'c', 'd'] -> [['a', 'b'], ['c', 'd']]
# example: ['a', 'b', 'c', 'd', 'e', 'f'] -> [['a', 'b'], ['c', 'd'], ['e', 'f']]
def grouper(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

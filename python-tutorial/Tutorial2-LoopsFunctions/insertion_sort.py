def insertion_sort(list):
    for index in range (1,len(list)):
        value = list[index]
        i = index -1
        while i>=0:
            if value < list[i]:
                # shift number in slot i right to slot i+1
                list[i+1] = list[i]
                # shift value left into slot i
                list[i] = value
                i = i-1
            else:
                break
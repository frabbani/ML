import random
from typing import List


# return 3 values a, b, and c, where a contains prize, b and c are empty
def choose_doors() -> List[int]:
    opt_a = random.randint(0, 2)
    opt_b = -1
    opt_c = -1
    if opt_a == 0:
        opt_b = 1
        opt_c = 2
    if opt_a == 1:
        opt_b = 0
        opt_c = 2
    if opt_a == 2:
        opt_b = 0
        opt_c = 1
    return [opt_a, opt_b, opt_c]


N = 50000   # run the simulation 50,000 times
first_selection_correct_count = 0   # collect count of the # of times the 1st selection contained the prize
second_selection_correct_count = 0  # collect count of the # of times the 2nd selection contained the prize
host_selection_correct_count = 0    # collect count the # of times the host opened door with prize (0)

for i in range(N):
    doors = ['?', '?', '?']
    abc = choose_doors()
    a = abc[0]
    b = abc[1]
    c = abc[2]
    bc = [b, c]
    doors[a] = 'prize'  # door a - always contains the prize
    doors[b] = 'empty'  # door b
    doors[c] = 'empty'  # door c
    host_opens_empty = -1  # since host always opens empty door, it is either b or c
    hidden_door = -1  # the door not opened, i.e. the second selection

    first_selection = random.randint(0, 2)  # user select
    if doors[first_selection] == 'prize':  # user selected a, so open b or c
        host_opens_empty = bc[random.randint(0, 1)]  # host selects random empty door (b or c)
        hidden_door = b if host_opens_empty == c else c
    if doors[first_selection] != 'prize':  # user selected b or c, so don't open a
        if first_selection == c:
            host_opens_empty = b
            hidden_door = a
        if first_selection == b:
            host_opens_empty = c
            hidden_door = a

    if doors[first_selection] == 'prize':
        first_selection_correct_count = first_selection_correct_count + 1
    if doors[hidden_door] == 'prize':
        second_selection_correct_count = second_selection_correct_count + 1
    if doors[host_opens_empty] == 'prize':
        host_selection_correct_count = host_selection_correct_count + 1

percentage = (first_selection_correct_count / N) * 100
print(f"first choice was correct {percentage:.2f}% of the time")
percentage = (second_selection_correct_count / N) * 100
print(f"second choice was correct {percentage:.2f}% of the time")
percentage = (host_selection_correct_count / N) * 100
print(f"host open the door with the prize {percentage:.2f}% of the time")

print("goodbye!")

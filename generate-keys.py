import random

AMOUNT = 8
LENGTH = 19

if __name__ == "__main__":
    s = '{{:0{}b}}'
    s = s.format(LENGTH)

    keys = []

    for i in range(AMOUNT):
        keys.append(s.format(random.getrandbits(LENGTH)))

    print(keys)

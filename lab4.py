import math
zadanie = 5

if zadanie == 1:

    def natural_numbers():
       num = 1
       while num <= 50:
           yield num
           num += 1
    print(natural_numbers())

    def is_prime(n:int):
        if n <= 1:
            return False
        if n <= 3:
            return True
        for num in range(2, math.ceil(n/2), 1):
            if num*num < n:
                if n % num != 0:
                    return False
        return True

    def prime_generator():
        num = 2
        while True:
            if is_prime(num):
                yield num
            num += 1

    if __name__ == "__main__":
        generator = prime_generator()
        for _ in range(10):
            print(next(generator))


if zadanie == 2:

    def multiplier(n):
        return lambda x: x * n

    multiply_by_3 = multiplier(3)
    result = multiply_by_3(5)
    print(result)


if zadanie == 3:
    values = [1, 2, 3, 4]
    def cube(val):
        return val * val * val

    result = list(map(cube, values))
    print(result)

if zadanie == 4:
    def even(n):
        return n % 2 == 0

    a = [1, 2, 3, 4, 5, 6]
    b = filter(even, a)
    print(list(b))

if zadanie == 5:
    
"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    count = 0 
    text = text.lower()
    for letters in text:
        if letters in "aeiou":
            count+=1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    letters = []
    for characters in text:
        if characters not in letters:
            letters.append(characters)
    if len(letters) == len(text):
        return True
    else:
        return False

def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    number = int(number)
    bin_number = bin(number)[2:]
    count = 0
    for character in bin_number:
        if int(character) == 1:
            count+=1
    return count 

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    number = int(number)
    while number > 9:
        new_number = 1
        for character in str(number):
            new_number *= int(character)
        number = new_number
        count+=1
    return count

def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    dlina = len(predicted)
    r = 0
    for i in range(dlina):
        r += (predicted[i] - expected[i])**2
    return (r / dlina)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    b = []
    number = int(number)
    for i in range(2,int(number**0.5) + 1):
        while number % i == 0:
            b.append(i)
            number = number // i
    if number > 1:
        b.append(number)
    result = ""
    k = 1
    j = 0
    while j < len(b):
        k = b.count(b[j])
        result += "(" + str(b[j]) 
        if k > 1:
            result += "**" + str(k) + ")"
        else:
            result += ')'
        j+=k
    return result

def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 0
    r = 1
    answer = "It is impossible"
    while cube_count > k:
        k += r**2
        r+=1
        if cube_count == k:
            answer = r-1
            break
    return answer




def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    sum1 = 0
    sum2 = 0
    dlina = len(str(number))
    if dlina%2 == 0:
        for i in range(dlina//2-1):
            sum1+=int(str(number)[i])
            sum2+=int(str(number)[-i-1])
    else:
        for i in range(dlina//2):
            sum1+=int(str(number)[i])
            sum2+=int(str(number)[-i-1])
    return sum1 == sum2
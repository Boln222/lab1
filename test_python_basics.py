import pytest
from grader_contracts.python_basics import TextInput, VectorPairInput, PositiveIntegerInput
from python_basics import count_vowels, has_unique_characters, count_one_bits, multiplicative_persistence
from python_basics import mse, prime_factorization, pyramid,is_balanced_number

def test_count_vowels_simple_word():
    assert count_vowels(TextInput(value="Hello")) == 2


def test_count_vowels_uppercase():
    assert count_vowels(TextInput(value="AEIOU")) == 5


def test_count_vowels_mixed_case():
    assert count_vowels(TextInput(value="HeLLo")) == 2


def test_count_vowels_empty_string():
    assert count_vowels(TextInput(value="")) == 0


def test_count_vowels_no_vowels():
    assert count_vowels(TextInput(value="xyz")) == 0


def test_count_vowels_single_letter():
    assert count_vowels(TextInput(value="a")) == 1


def test_count_vowels_repeated():
    assert count_vowels(TextInput(value="aaa")) == 3


def test_has_unique_characters_1():
    assert has_unique_characters(TextInput(value="abcde")) == True

def test_has_unique_characters_2():
    assert has_unique_characters(TextInput(value="aabcddee")) == False

def test_count_one_bits_1():
    assert count_one_bits(TextInput(value="125")) == 6

def test_multiplicative_persistence_1():
    assert multiplicative_persistence(TextInput(value="39")) == 3

def test_multiplicative_persistence_2():
    assert multiplicative_persistence(TextInput(value="4")) == 0

def test_multiplicative_persistence_3():
    assert multiplicative_persistence(TextInput(value="999")) == 4

def test_mse_1():
    result = mse(VectorPairInput(predicted = [1,2,3], expected = [1,2,5]))
    assert result == pytest.approx(4/3)

def test_prime_factorization_1():
    assert prime_factorization(PositiveIntegerInput(value = "86240")) == "(2**5)(5)(7**2)(11)"

def test_pyramid_1():
    assert pyramid(PositiveIntegerInput(value = 55)) == 5

def test_pyramid_2():
    assert pyramid(PositiveIntegerInput(value = 60313)) == "It is impossible"

def test_is_balanced_number_1():
    assert is_balanced_number(PositiveIntegerInput(value = 1234006)) == True

def test_is_balanced_number_2():
    assert is_balanced_number(PositiveIntegerInput(value = 123456))== False
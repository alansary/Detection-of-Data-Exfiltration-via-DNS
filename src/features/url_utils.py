"""Utilities for processing URLs."""

import numpy as np
from collections import Counter
import math

def get_FQDN_count(url):
  """Returns the number of characters in FQDN as an integer.

  Args:
    url: The url whose characters are to be counted.
  """
  return len(''.join(url.split('.')))

def get_subdomain_length(url):
  """Returns the length of subdomain in FQDN as an integer.

  Args:
    url: The url whose subdomain length is to be counted.
  """
  return len(' '.join(url.split('.')[:-2]))

def get_digit_count(word):
  """Returns the number of digits in the word as an integer.

  Args:
    word: The string whose digits are to be counted.
  """
  count = 0
  for char in word:
    if char.isdigit():
      count += 1
  return count

def get_uppercase_characters_count(word):
  """Returns the number of uppercase characters in the word as an integer.

  Args:
    word: The string whose uppercase characters are to be counted.
  """
  count = 0
  for char in word:
    if char.isalpha() and char.isupper():
      count += 1
  return count

def get_lowercase_characters_count(word):
  """Returns the number of lowercase characters in the word as an integer.

  Args:
    word: The string whose lowercase characters are to be counted.
  """
  count = 0
  for char in word:
    if char.isalpha() and char.islower():
      count += 1
  return count

def is_special_character(char):
  """Returns True if the character is an ASCII special character.

  The special characters can be found here:
  https://en.wikipedia.org/wiki/List_of_Unicode_characters#Latin_script.

  Args:
    char: The character to consider.
  """
  char_int = ord(char)
  return ((32 <= char_int <= 47) or (58 <= char_int <= 64) or
          (91 <= char_int <= 96) or (123 <= char_int <= 126))

def get_special_characters_count(url):
  """Returns the number of special characters in the url as an integer.

  Args:
    url: The url whose special characters are to be counted.
  """
  url = ''.join(url.split('.'))
  count = 0
  for char in url:
    if is_special_character(char):
      count += 1
  return count

def get_url_labels_count(url):
  """Returns the number of labels in the url as an integer.

  Args:
    url: The url whose labels are to be counted.
  """
  return len(url.split('.'))

def get_max_label_length(url):
  """Returns the maximum label length in the url as an integer.

  Args:
    url: The url whose maximum label length is to be computed.
  """
  return max([len(label) for label in url.split('.')])

def get_avg_label_length(url):
  """Returns the average label length in the url as an integer.

  Args:
    url: The url whose average label length is to be computed.
  """
  return np.mean([len(label) for label in url.split('.')])

def get_sld_length(url):
  """Returns the length of sld in FQDN as an integer.

  Args:
    url: The url whose sld length is to be counted.
  """
  return len(url.split('.')[-2]) if len(url.split('.')) >= 2 else len(url)

def get_length_of_domain_and_subdomain_length(url):
  """Returns the length of domain and subdomain in FQDN as an integer.

  Args:
    url: The url whose domain and subdomain length is to be counted.
  """
  return len(''.join(url.split('.')[:-1]))

def url_has_subdomain(url):
  """Returns whether the url has subdomain or not.

  Args:
    url: The url we will check a subdomain for.
  """
  return 1 if len(url.split('.')) > 2 else 0

def get_longest_word(url):
  """Returns the longest word in url.
  
  Args:
    url: The url we will get the longest word for.
  """
  words = url.split('.')
  counts = [len(word) for word in words]
  return words[np.argmax(counts)]

def entropy(url, base=2):
  """Returns the Shannon entropy of url.
  
  Args:
    url: The url we will the Shannon entropy for.
  """
  entropy = 0.0

  if len(url) > 0:
      cnt = Counter(url)
      length = len(url)
      for count in cnt.values():
          entropy += (count / length) * math.log(count / length, base)
      entropy = entropy * -1.0
  
  return entropy

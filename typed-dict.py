from typing import TypedDict


class person(TypedDict):
    name : str
    age : int

new_person = person = {'name': 'Vatsal', 'age':22}

print(new_person)
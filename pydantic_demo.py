from pydantic import BaseModel,EmailStr
from typing import Optional

class Student(BaseModel):
    name : str = 'vatsal'
    age : Optional [int] = None
    email : EmailStr

new_student = {'age':'22', 'email':'vatsal@example.com'}  # this will work because pydantic will try to convert the string '22' to an integer 22

student = Student(**new_student)

print(student)


import logging, unittest, enum
'''
class bul_types(enum.Enum):
    def __init__(self):
        SPIRAL =0
        aimed=1


class Foo():

    def __init__(self):
        self.is_weird = True

    @property
    def is_not_weird(self):
        return 4

    def do_shit(self):
        self.is_weird = False



def bar(lst=None):
    if lst is None:
        lst = []
    lst.append(1)
    print(lst)


def identity(x: int) -> int:
    return x


class simpler_obj:
    def do_thing(self, x):
        return x ** 2


class parent(simpler_obj):
    def __init__(self):
        print("foo")


def square(n):
    return n**2+1
class tester(unittest.TestCase):
    def test_case_square(self):
        assert( square(0)==0), "Test case failed for input 0"



logging.basicConfig(filename="test.log", level=logging.INFO)
logging.info("testing")
logging.info([1, 2, 3])
bar()
bar()
bar()
print(identity("a"))
'''



def do_twice(func):
    def wrapper_do_twice(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)
    return wrapper_do_twice

@do_twice
def print_once(input):
    print(f"printing this twice{input}")
    return "asdf"



def p_decorate(func):
   def func_wrapper(name):
       return "<p>{0}</p>".format(func(name))
   return func_wrapper

@p_decorate
def get_text(name):
   return "lorem ipsum, {0} dolor sit amet".format(name)

print_once("f00")
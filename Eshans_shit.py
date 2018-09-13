class Foo():

    def __init__(self):
        self.is_weird = True

    #'@property
    def is_not_weird(self, num_input):
        print(num_input*2)
        return not self.is_weird

    def do_shit(self):
        self.is_weird = False

def bar(lst=None):
    if lst is None:
        lst = []
    lst.append(1)
    print(lst)

if __name__ == "__main__":

    bar()
    bar()
    bar()

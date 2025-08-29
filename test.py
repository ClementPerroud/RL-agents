class First(object):
    def __init__(self, var1, **kwargs):
        super().__init__()
        self.var1 = var1
        print("first", self.var1)

class Second(First):
    def __init__(self, var1, var2, **kwargs):
        super().__init__(var1=var1, **kwargs)
        self.var2 = var2
        print("second", self.var2)

class Third(First):
    def __init__(self, var1, var3, **kwargs):
        super().__init__(var1=var1, **kwargs)
        self.var3 = var3
        print("third", self.var3)

class Fourth(Second, Third):
    def __init__(self, var1, var2, var3, var4):
        self.var4 = var4
        super().__init__(var1=var1, var2=var2, var3=var3)
        print("that's it", var4)

foo = Fourth("1", "2", "3", "4")
print(foo.var1, foo.var2, foo.var3, foo.var4)
